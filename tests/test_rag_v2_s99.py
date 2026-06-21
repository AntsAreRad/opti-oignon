"""
Tests for S99 -- RAG v2: Vector Store + Document Ingestion.

Validates:
- Part 1: RAGChunker (plain text, markdown, CSV, code, PDF text, overlap, metadata)
- Part 2: RAGVectorStore (SQLite DB, collections CRUD, documents, citations, query graceful)
- Part 3: Web ingestion (HTML extraction, config loader)
- Part 4: API routes (schemas, endpoints, wiring)
- Part 5: Frontend (types, API client, KnowledgeBasePanel, settings tab)
- Part 6: Config (rag.yaml)
- Part 7: Integration wiring (deps.py, app.py, version bump)
- Zero regressions

Target: ~55 tests
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
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
BACKEND_DIR = os.path.join(PROJECT_ROOT, "opti_oignon")
API_DIR = os.path.join(BACKEND_DIR, "api")
CONFIG_DIR = os.path.join(BACKEND_DIR, "config")
FRONTEND_SRC = os.path.join(PROJECT_ROOT, "frontend", "src")
COMPONENTS_DIR = os.path.join(FRONTEND_SRC, "lib", "components")
ROUTES_DIR = os.path.join(FRONTEND_SRC, "routes")
API_TS_DIR = os.path.join(FRONTEND_SRC, "lib", "api")
TYPES_TS = os.path.join(FRONTEND_SRC, "lib", "types.ts")


def _load_module(name, path):
    """Load a Python module from file path without triggering __init__ imports."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # Stub parent packages
    if "opti_oignon" not in sys.modules:
        parent = type(sys)("opti_oignon")
        sys.modules["opti_oignon"] = parent
    if "opti_oignon.config" not in sys.modules:
        cfg_mod = type(sys)("opti_oignon.config")
        cfg_mod.DATA_DIR = tempfile.mkdtemp()
        sys.modules["opti_oignon.config"] = cfg_mod
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _read(path):
    """Read file contents as string."""
    with open(path, encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Load modules
# ---------------------------------------------------------------------------

_chunker_mod = _load_module(
    "opti_oignon.rag_chunker",
    os.path.join(BACKEND_DIR, "rag_chunker.py"),
)
RAGChunker = _chunker_mod.RAGChunker
RAGChunk = _chunker_mod.RAGChunk
ChunkingResult = _chunker_mod.ChunkingResult

# Need rag_chunker in sys.modules before rag_store imports it
_store_mod = _load_module(
    "opti_oignon.rag_store",
    os.path.join(BACKEND_DIR, "rag_store.py"),
)
RAGVectorStore = _store_mod.RAGVectorStore
_RAGDatabase = _store_mod._RAGDatabase
IngestedDocument = _store_mod.IngestedDocument
CollectionInfo = _store_mod.CollectionInfo
CitationRecord = _store_mod.CitationRecord
RetrievalResult = _store_mod.RetrievalResult
QueryResponse = _store_mod.QueryResponse
load_rag_config = _store_mod.load_rag_config


# =============================================================================
# PART 1: RAGChunker
# =============================================================================


class TestRAGChunkerPlainText(unittest.TestCase):
    """Test plain text chunking."""

    def setUp(self):
        self.chunker = RAGChunker(chunk_size=100, chunk_overlap=0)

    def test_chunk_text_produces_chunks(self):
        text = "First paragraph about ecology.\n\nSecond paragraph about biodiversity.\n\nThird about Shannon index."
        result = self.chunker.chunk_text(text, source="test.txt")
        self.assertIsInstance(result, ChunkingResult)
        self.assertGreater(result.chunk_count, 0)

    def test_chunk_text_metadata(self):
        text = "Hello world. " * 50
        result = self.chunker.chunk_text(text, source="test.txt", file_type="text")
        for chunk in result.chunks:
            meta = chunk.metadata
            self.assertEqual(meta["source_file"], "test.txt")
            self.assertEqual(meta["file_type"], "text")
            self.assertIn("parent_doc_id", meta)
            self.assertIn("position", meta)
            self.assertIn("char_count", meta)

    def test_chunk_text_empty_input(self):
        result = self.chunker.chunk_text("", source="empty.txt")
        self.assertEqual(result.chunk_count, 0)

    def test_chunk_text_short_input(self):
        result = self.chunker.chunk_text("tiny", source="short.txt")
        self.assertEqual(result.chunk_count, 0)  # < 10 chars

    def test_chunk_id_deterministic(self):
        result = self.chunker.chunk_text("Some content for testing chunks here.", source="a.txt", doc_id="fixed123")
        chunk = result.chunks[0]
        cid1 = chunk.chunk_id
        # Same doc_id and chunk_index should produce same chunk_id
        result2 = self.chunker.chunk_text("Different content entirely.", source="b.txt", doc_id="fixed123")
        cid2 = result2.chunks[0].chunk_id
        self.assertEqual(cid1, cid2)  # Same doc_id + index 0

    def test_doc_id_generated_if_absent(self):
        result = self.chunker.chunk_text("Some content for testing doc ID generation.", source="a.txt")
        self.assertTrue(len(result.doc_id) > 0)


class TestRAGChunkerMarkdown(unittest.TestCase):
    """Test markdown chunking by headers."""

    def setUp(self):
        self.chunker = RAGChunker(chunk_size=200, chunk_overlap=0)

    def test_markdown_splits_by_headers(self):
        md = "# Intro\nFirst section content here.\n\n## Methods\nSecond section about methods.\n\n## Results\nThird section with results."
        result = self.chunker.chunk_text(md, source="paper.md", file_type="markdown")
        sections = [c.section for c in result.chunks]
        self.assertTrue(any("Intro" in (s or "") for s in sections))
        self.assertTrue(any("Methods" in (s or "") for s in sections))

    def test_markdown_no_headers_fallback(self):
        text = "Just plain text without any markdown headers at all. " * 10
        result = self.chunker.chunk_text(text, source="plain.md", file_type="markdown")
        self.assertGreater(result.chunk_count, 0)


class TestRAGChunkerCSV(unittest.TestCase):
    """Test CSV chunking with header preservation."""

    def setUp(self):
        self.chunker = RAGChunker(chunk_size=50, chunk_overlap=0)

    def test_csv_keeps_header(self):
        csv = "name,value,site\n" + "\n".join(
            f"species_{i},{i*10},BCI" for i in range(20)
        )
        result = self.chunker.chunk_text(csv, source="data.csv", file_type="csv")
        self.assertGreater(result.chunk_count, 0)
        # Each chunk should start with the header
        for chunk in result.chunks:
            self.assertTrue(chunk.content.startswith("name,value,site"))

    def test_csv_section_naming(self):
        csv = "col1,col2\nA,1\nB,2\nC,3"
        result = self.chunker.chunk_text(csv, source="small.csv", file_type="csv")
        self.assertTrue(any("rows_" in (c.section or "") for c in result.chunks))


class TestRAGChunkerCode(unittest.TestCase):
    """Test code chunking by function/class boundaries."""

    def setUp(self):
        self.chunker = RAGChunker(chunk_size=200, chunk_overlap=0)

    def test_code_splits_by_function(self):
        code = textwrap.dedent("""\
            def hello(name):
                return f"Hello {name}"

            def goodbye(name):
                return f"Goodbye {name}"

            class Greeter:
                def __init__(self, name):
                    self.name = name
        """)
        result = self.chunker.chunk_text(code, source="main.py", file_type="python")
        sections = [c.section for c in result.chunks]
        self.assertTrue(any("hello" in (s or "") for s in sections))
        self.assertTrue(any("Greeter" in (s or "") for s in sections))


class TestRAGChunkerOverlap(unittest.TestCase):
    """Test overlap injection between chunks."""

    def test_overlap_adds_context(self):
        chunker = RAGChunker(chunk_size=50, chunk_overlap=20)
        text = "Alpha paragraph one.\n\nBeta paragraph two.\n\nGamma paragraph three.\n\nDelta paragraph four."
        result = chunker.chunk_text(text, source="test.txt")
        if result.chunk_count > 1:
            # Middle chunks should contain overlap markers
            found_overlap = False
            for chunk in result.chunks[1:]:
                if "..." in chunk.content:
                    found_overlap = True
                    break
            self.assertTrue(found_overlap, "Expected overlap markers in chunks")

    def test_zero_overlap(self):
        chunker = RAGChunker(chunk_size=50, chunk_overlap=0)
        text = "A sentence here.\n\nAnother one.\n\nThird paragraph."
        result = chunker.chunk_text(text, source="test.txt")
        for chunk in result.chunks:
            self.assertNotIn("...\n", chunk.content)


class TestRAGChunkerFile(unittest.TestCase):
    """Test chunk_file with real temp files."""

    def test_chunk_text_file(self):
        chunker = RAGChunker(chunk_size=100, chunk_overlap=0)
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False, encoding="utf-8") as f:
            f.write("This is a test document.\n\nIt has multiple paragraphs.\n\nThird paragraph here.")
            f.flush()
            result = chunker.chunk_file(f.name)
        os.unlink(f.name)
        self.assertGreater(result.chunk_count, 0)
        self.assertEqual(result.extraction_method, "text")

    def test_chunk_missing_file(self):
        chunker = RAGChunker()
        with self.assertRaises(FileNotFoundError):
            chunker.chunk_file("/nonexistent/file.txt")

    def test_ext_mapping(self):
        self.assertEqual(RAGChunker.EXT_MAP.get(".pdf"), "pdf")
        self.assertEqual(RAGChunker.EXT_MAP.get(".docx"), "docx")
        self.assertEqual(RAGChunker.EXT_MAP.get(".csv"), "csv")
        self.assertEqual(RAGChunker.EXT_MAP.get(".py"), "python")
        self.assertEqual(RAGChunker.EXT_MAP.get(".md"), "markdown")


class TestRAGChunkDataclass(unittest.TestCase):
    """Test RAGChunk properties."""

    def test_metadata_keys(self):
        chunk = RAGChunk(
            content="test",
            source_file="a.txt",
            file_type="text",
            chunk_index=0,
            total_chunks=1,
            parent_doc_id="doc1",
            position=0,
        )
        meta = chunk.metadata
        expected_keys = {
            "source_file", "file_type", "chunk_index", "total_chunks",
            "parent_doc_id", "position", "page", "section",
            "start_line", "end_line", "char_count",
        }
        self.assertEqual(set(meta.keys()), expected_keys)

    def test_repr(self):
        chunk = RAGChunk(
            content="Hello world",
            source_file="/path/to/file.txt",
            file_type="text",
            chunk_index=0,
            total_chunks=5,
            parent_doc_id="doc1",
        )
        r = repr(chunk)
        self.assertIn("RAGChunk", r)
        self.assertIn("file.txt", r)


# =============================================================================
# PART 2: RAGVectorStore + SQLite
# =============================================================================


class TestRAGDatabase(unittest.TestCase):
    """Test the SQLite backing store."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db = _RAGDatabase(os.path.join(self.tmpdir, "test.db"))

    def test_create_and_list_collection(self):
        self.db.create_collection("papers", "Research papers")
        colls = self.db.list_collections()
        self.assertEqual(len(colls), 1)
        self.assertEqual(colls[0]["name"], "papers")
        self.assertEqual(colls[0]["description"], "Research papers")

    def test_delete_collection_cascades(self):
        self.db.create_collection("temp")
        doc = IngestedDocument(
            doc_id="d1", collection_name="temp", source_file="x.pdf",
            file_type="pdf", chunk_count=3, raw_text_length=100,
            ingested_at=time.time(),
        )
        self.db.insert_document(doc)
        cit = CitationRecord(
            citation_id="c1", query="test", collection_name="temp",
            chunk_id="ch1", parent_doc_id="d1", source_file="x.pdf",
            section="intro", score=0.9, timestamp=time.time(),
        )
        self.db.insert_citation(cit)
        self.db.delete_collection("temp")
        self.assertEqual(len(self.db.list_collections()), 0)
        self.assertEqual(len(self.db.list_documents()), 0)
        self.assertEqual(len(self.db.list_citations()), 0)

    def test_insert_and_get_document(self):
        self.db.create_collection("coll")
        doc = IngestedDocument(
            doc_id="d2", collection_name="coll", source_file="file.txt",
            file_type="text", chunk_count=5, raw_text_length=500,
            ingested_at=time.time(), metadata={"key": "value"},
        )
        self.db.insert_document(doc)
        retrieved = self.db.get_document("d2")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.source_file, "file.txt")
        self.assertEqual(retrieved.metadata["key"], "value")

    def test_get_nonexistent_document(self):
        result = self.db.get_document("nonexistent")
        self.assertIsNone(result)

    def test_delete_document(self):
        self.db.create_collection("coll")
        doc = IngestedDocument(
            doc_id="d3", collection_name="coll", source_file="f.csv",
            file_type="csv", chunk_count=2, raw_text_length=200,
            ingested_at=time.time(),
        )
        self.db.insert_document(doc)
        info = self.db.delete_document("d3")
        self.assertIsNotNone(info)
        self.assertEqual(len(self.db.list_documents()), 0)

    def test_citation_insert_and_list(self):
        self.db.create_collection("coll")
        for i in range(3):
            cit = CitationRecord(
                citation_id=f"c{i}", query=f"query {i}", collection_name="coll",
                chunk_id=f"ch{i}", parent_doc_id="d1", source_file="f.txt",
                section=None, score=0.5 + i * 0.1, timestamp=time.time(),
            )
            self.db.insert_citation(cit)
        all_cits = self.db.list_citations(collection_name="coll")
        self.assertEqual(len(all_cits), 3)

    def test_touch_collection(self):
        self.db.create_collection("coll")
        colls_before = self.db.list_collections()
        t1 = colls_before[0]["updated_at"]
        time.sleep(0.05)
        self.db.touch_collection("coll")
        colls_after = self.db.list_collections()
        t2 = colls_after[0]["updated_at"]
        self.assertGreater(t2, t1)


class TestRAGVectorStoreGraceful(unittest.TestCase):
    """Test RAGVectorStore without ChromaDB."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = RAGVectorStore(data_dir=os.path.join(self.tmpdir, "rag"))

    def test_query_returns_empty(self):
        resp = self.store.query("anything")
        self.assertIsInstance(resp, QueryResponse)
        self.assertEqual(resp.total_results, 0)

    def test_list_documents_empty(self):
        docs = self.store.list_documents()
        self.assertEqual(len(docs), 0)

    def test_list_collections_empty(self):
        colls = self.store.list_collections()
        self.assertEqual(len(colls), 0)

    def test_list_citations_empty(self):
        cits = self.store.list_citations()
        self.assertEqual(len(cits), 0)

    def test_delete_nonexistent_doc(self):
        result = self.store.delete_document("nope")
        self.assertFalse(result)


class TestDataclassSerialization(unittest.TestCase):
    """Test to_dict on all data classes."""

    def test_collection_info_to_dict(self):
        c = CollectionInfo("test", "desc", 3, 10, 1.0, 2.0)
        d = c.to_dict()
        self.assertEqual(d["name"], "test")
        self.assertEqual(d["document_count"], 3)

    def test_retrieval_result_to_dict(self):
        r = RetrievalResult("content", 0.85, "file.pdf", "pdf", 0, 5, "doc1", "coll")
        d = r.to_dict()
        self.assertEqual(d["score"], 0.85)
        self.assertEqual(d["parent_doc_id"], "doc1")

    def test_citation_to_dict(self):
        c = CitationRecord("c1", "query", "coll", "ch1", "d1", "f.txt", "intro", 0.9, 1.0)
        d = c.to_dict()
        self.assertEqual(d["citation_id"], "c1")

    def test_query_response_to_dict(self):
        qr = QueryResponse("q", [], [], 0)
        d = qr.to_dict()
        self.assertEqual(d["query"], "q")
        self.assertEqual(d["total_results"], 0)

    def test_ingested_document_to_dict(self):
        doc = IngestedDocument("d1", "coll", "f.pdf", "pdf", 5, 1000, 1.0)
        d = doc.to_dict()
        self.assertEqual(d["doc_id"], "d1")


# =============================================================================
# PART 3: Web Ingestion + Config
# =============================================================================


class TestHTMLExtraction(unittest.TestCase):
    """Test HTML text extraction and cleaning."""

    def _extract(self, html, cfg=None):
        if cfg is None:
            cfg = {
                "strip_tags": ["nav", "header", "footer", "aside", "script", "style"],
                "boilerplate_patterns": ["sidebar", "menu", "navbar", "cookie"],
            }
        return RAGVectorStore._extract_html_text(html, cfg)

    def test_strips_script_tags(self):
        html = "<p>Hello</p><script>var x = 1;</script><p>World</p>"
        text = self._extract(html)
        self.assertNotIn("var x", text)
        self.assertIn("Hello", text)
        self.assertIn("World", text)

    def test_strips_nav(self):
        html = "<nav><ul><li>Home</li></ul></nav><main><p>Content here</p></main>"
        text = self._extract(html)
        self.assertNotIn("Home", text)
        self.assertIn("Content", text)

    def test_strips_boilerplate_classes(self):
        html = '<div class="sidebar-widget">Ads</div><p>Real content</p>'
        text = self._extract(html)
        self.assertIn("Real content", text)

    def test_decodes_entities(self):
        html = "<p>Tom &amp; Jerry &lt;3</p>"
        text = self._extract(html)
        self.assertIn("Tom & Jerry", text)
        self.assertIn("<3", text)

    def test_collapses_whitespace(self):
        html = "<p>Hello</p>   \n\n\n   <p>World</p>"
        text = self._extract(html)
        # Should not have excessive blank lines
        self.assertNotIn("\n\n\n", text)


class TestRAGYAMLConfig(unittest.TestCase):
    """Test rag.yaml configuration file."""

    def test_yaml_exists(self):
        path = os.path.join(CONFIG_DIR, "rag.yaml")
        self.assertTrue(os.path.exists(path), "config/rag.yaml must exist")

    def test_yaml_parses(self):
        path = os.path.join(CONFIG_DIR, "rag.yaml")
        with open(path) as f:
            cfg = yaml.safe_load(f)
        self.assertIsInstance(cfg, dict)

    def test_yaml_sections(self):
        path = os.path.join(CONFIG_DIR, "rag.yaml")
        with open(path) as f:
            cfg = yaml.safe_load(f)
        for key in ("chunking", "embedding", "retrieval", "web_ingestion", "collections", "storage"):
            self.assertIn(key, cfg, f"Missing section: {key}")

    def test_yaml_embedding_model(self):
        path = os.path.join(CONFIG_DIR, "rag.yaml")
        with open(path) as f:
            cfg = yaml.safe_load(f)
        self.assertEqual(cfg["embedding"]["model"], "mxbai-embed-large")

    def test_yaml_web_ingestion_tags(self):
        path = os.path.join(CONFIG_DIR, "rag.yaml")
        with open(path) as f:
            cfg = yaml.safe_load(f)
        tags = cfg["web_ingestion"]["strip_tags"]
        self.assertIn("script", tags)
        self.assertIn("nav", tags)
        self.assertIn("footer", tags)

    def test_load_rag_config_function(self):
        cfg = load_rag_config()
        self.assertIn("chunking", cfg)
        self.assertIn("embedding", cfg)
        self.assertEqual(cfg["chunking"]["chunk_size"], 500)


# =============================================================================
# PART 4: API Routes
# =============================================================================


class TestRAGRoutesFile(unittest.TestCase):
    """Test routes_rag.py structure and schemas."""

    def test_file_exists(self):
        path = os.path.join(API_DIR, "routes_rag.py")
        self.assertTrue(os.path.exists(path))

    def test_ast_parses(self):
        path = os.path.join(API_DIR, "routes_rag.py")
        src = _read(path)
        tree = ast.parse(src)
        self.assertIsNotNone(tree)

    def test_has_router(self):
        src = _read(os.path.join(API_DIR, "routes_rag.py"))
        self.assertIn('APIRouter(prefix="/api/rag"', src)

    def test_endpoint_count(self):
        src = _read(os.path.join(API_DIR, "routes_rag.py"))
        # Count @router.get/post/delete decorators
        # 8 original (S99) + 5 batch ingestion (S119)
        endpoints = re.findall(r'@router\.(get|post|delete|put|patch)\(', src)
        self.assertGreaterEqual(len(endpoints), 13, f"Expected >=13 endpoints, found {len(endpoints)}")

    def test_schemas_defined(self):
        src = _read(os.path.join(API_DIR, "routes_rag.py"))
        for schema in (
            "CollectionCreateRequest", "CollectionResponse",
            "IngestResponse", "IngestURLRequest",
            "QueryRequest", "QueryResponseSchema",
            "DocumentResponse", "DocumentDeleteResponse",
        ):
            self.assertIn(f"class {schema}", src, f"Missing schema: {schema}")

    def test_endpoints_present(self):
        src = _read(os.path.join(API_DIR, "routes_rag.py"))
        for route in (
            '"/collections"',
            '"/ingest"',
            '"/ingest/url"',
            '"/query"',
            '"/documents"',
        ):
            self.assertIn(route, src, f"Missing route: {route}")

    def test_no_french(self):
        src = _read(os.path.join(API_DIR, "routes_rag.py"))
        for word in ("Erreur", "Aucun", "Recherche", "Supprimer"):
            self.assertNotIn(word, src, f"French word found: {word}")


# =============================================================================
# PART 5: Frontend
# =============================================================================


class TestFrontendTypes(unittest.TestCase):
    """Test TypeScript types for RAG."""

    def test_types_file_has_rag_interfaces(self):
        src = _read(TYPES_TS)
        for iface in (
            "RAGCollection", "RAGCollectionsListResponse",
            "RAGDocument", "RAGDocumentsListResponse",
            "RAGIngestResponse", "RAGIngestURLRequest",
            "RAGQueryRequest", "RAGQueryResponse",
            "RAGRetrievalResult", "RAGCitation",
        ):
            self.assertIn(f"export interface {iface}", src, f"Missing interface: {iface}")


class TestFrontendAPIClient(unittest.TestCase):
    """Test rag.ts API client."""

    def test_file_exists(self):
        path = os.path.join(API_TS_DIR, "rag.ts")
        self.assertTrue(os.path.exists(path))

    def test_exports_functions(self):
        src = _read(os.path.join(API_TS_DIR, "rag.ts"))
        for fn in (
            "listCollections", "createCollection", "deleteCollection",
            "ingestFile", "ingestURL",
            "queryKnowledgeBase",
            "listDocuments", "deleteDocument",
        ):
            self.assertIn(f"export async function {fn}", src, f"Missing export: {fn}")

    def test_uses_correct_endpoints(self):
        src = _read(os.path.join(API_TS_DIR, "rag.ts"))
        self.assertIn("/api/rag/collections", src)
        self.assertIn("/api/rag/ingest", src)
        self.assertIn("/api/rag/ingest/url", src)
        self.assertIn("/api/rag/query", src)
        self.assertIn("/api/rag/documents", src)

    def test_imports_from_client(self):
        src = _read(os.path.join(API_TS_DIR, "rag.ts"))
        self.assertIn("from './client'", src)

    def test_no_hardcoded_hex(self):
        src = _read(os.path.join(API_TS_DIR, "rag.ts"))
        hex_matches = re.findall(r'#[0-9a-fA-F]{3,8}\b', src)
        self.assertEqual(len(hex_matches), 0, f"Hardcoded hex found: {hex_matches}")


class TestKnowledgeBasePanel(unittest.TestCase):
    """Test KnowledgeBasePanel.svelte."""

    def setUp(self):
        self.path = os.path.join(COMPONENTS_DIR, "settings", "KnowledgeBasePanel.svelte")
        self.src = _read(self.path)

    def test_file_exists(self):
        self.assertTrue(os.path.exists(self.path))

    def test_imports_api_functions(self):
        for fn in ("listCollections", "createCollection", "ingestFile", "ingestURL", "queryKnowledgeBase"):
            self.assertIn(fn, self.src, f"Missing import: {fn}")

    def test_has_subtabs(self):
        self.assertIn("collections", self.src)
        self.assertIn("documents", self.src)
        self.assertIn("query", self.src)

    def test_has_drag_drop(self):
        self.assertIn("handleDrop", self.src)
        self.assertIn("dragOver", self.src)

    def test_has_url_ingestion(self):
        self.assertIn("urlInput", self.src)
        self.assertIn("handleURLIngest", self.src)

    def test_has_citation_display(self):
        self.assertIn("citations", self.src)
        self.assertIn("chunk_id", self.src)

    def test_no_hardcoded_hex(self):
        # Extract style attributes and check for hex colors
        hex_in_styles = re.findall(r'style="[^"]*#[0-9a-fA-F]{3,8}[^"]*"', self.src)
        self.assertEqual(len(hex_in_styles), 0, f"Hardcoded hex in styles: {hex_in_styles}")

    def test_uses_css_variables(self):
        self.assertIn("var(--oo-", self.src)

    def test_no_french(self):
        for word in ("Aucun", "Erreur", "Charger", "Supprimer", "Recherche"):
            self.assertNotIn(word, self.src, f"French word found: {word}")


class TestSettingsPageKnowledgeTab(unittest.TestCase):
    """Test that settings page includes Knowledge tab."""

    def setUp(self):
        self.src = _read(os.path.join(ROUTES_DIR, "settings", "+page.svelte"))

    def test_imports_knowledge_panel(self):
        self.assertIn("KnowledgeBasePanel", self.src)

    def test_tab_type_includes_knowledge(self):
        self.assertIn("'knowledge'", self.src)

    def test_tab_entry_exists(self):
        self.assertIn("Knowledge", self.src)
        self.assertIn("knowledge base", self.src.lower())

    def test_renders_panel(self):
        self.assertIn("<KnowledgeBasePanel", self.src)


# =============================================================================
# PART 7: Integration Wiring
# =============================================================================


class TestDepsWiring(unittest.TestCase):
    """Test deps.py RAG integration."""

    def setUp(self):
        self.src = _read(os.path.join(API_DIR, "deps.py"))

    def test_rag_store_import(self):
        self.assertIn("RAG_STORE_AVAILABLE", self.src)
        self.assertIn("get_rag_store", self.src)

    def test_rag_chunker_import(self):
        self.assertIn("RAG_CHUNKER_AVAILABLE", self.src)


class TestAppWiring(unittest.TestCase):
    """Test app.py RAG integration."""

    def setUp(self):
        self.src = _read(os.path.join(API_DIR, "app.py"))

    def test_rag_router_imported(self):
        self.assertIn("from .routes_rag import router as rag_router", self.src)

    def test_rag_router_registered(self):
        self.assertIn("app.include_router(rag_router)", self.src)

    def test_version_bumped(self):
        self.assertIn('version="1.10.2"', self.src)

    def test_health_includes_rag(self):
        self.assertIn("RAG_STORE_AVAILABLE", self.src)
        self.assertIn("RAG_CHUNKER_AVAILABLE", self.src)
        self.assertIn('"rag_store"', self.src)
        self.assertIn('"rag_chunker"', self.src)


class TestBackendModulesAST(unittest.TestCase):
    """Verify all new Python files parse correctly."""

    def test_rag_chunker_ast(self):
        src = _read(os.path.join(BACKEND_DIR, "rag_chunker.py"))
        tree = ast.parse(src)
        classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        self.assertIn("RAGChunker", classes)
        self.assertIn("RAGChunk", classes)

    def test_rag_store_ast(self):
        src = _read(os.path.join(BACKEND_DIR, "rag_store.py"))
        tree = ast.parse(src)
        classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        self.assertIn("RAGVectorStore", classes)
        self.assertIn("_RAGDatabase", classes)

    def test_routes_rag_ast(self):
        src = _read(os.path.join(API_DIR, "routes_rag.py"))
        ast.parse(src)

    def test_deps_ast(self):
        src = _read(os.path.join(API_DIR, "deps.py"))
        ast.parse(src)

    def test_app_ast(self):
        src = _read(os.path.join(API_DIR, "app.py"))
        ast.parse(src)


class TestNoEmojisInCode(unittest.TestCase):
    """Verify no emojis in new Python files."""

    EMOJI_RE = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F900-\U0001F9FF"
        "\U00002702-\U000027B0"
        "]+",
        flags=re.UNICODE,
    )

    def _check_file(self, path):
        src = _read(path)
        matches = self.EMOJI_RE.findall(src)
        self.assertEqual(len(matches), 0, f"Emojis found in {path}: {matches}")

    def test_rag_chunker(self):
        self._check_file(os.path.join(BACKEND_DIR, "rag_chunker.py"))

    def test_rag_store(self):
        self._check_file(os.path.join(BACKEND_DIR, "rag_store.py"))

    def test_routes_rag(self):
        self._check_file(os.path.join(API_DIR, "routes_rag.py"))


if __name__ == "__main__":
    unittest.main()
