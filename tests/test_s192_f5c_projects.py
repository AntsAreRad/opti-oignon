#!/usr/bin/env python3
"""
S192 F5c tests -- projects / project_context / project_triggers
(PCX-01, PTR-01) plus RG-01 verification pins and recorded-finding pins.

Loaders pre-seed sys.modules with stub opti_oignon.* dependencies before
exec so the guarded absolute imports resolve to stubs and the module-level
singletons (ProjectStore()) never fire their import side effects
(creating data/*.db -- the P-01 trigger class, recorded as PRJ-03).
"""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
PKG_DIR = REPO_ROOT / "opti_oignon"


# =============================================================================
# Loaders (sys.modules pre-seed idiom)
# =============================================================================

def _preseed_opti_oignon_stubs():
    """Stub the opti_oignon package tree consumed by the F5c modules."""
    if "opti_oignon" in sys.modules and getattr(
        sys.modules["opti_oignon"], "_s192_stub", False
    ):
        return
    root = types.ModuleType("opti_oignon")
    root.__path__ = []
    root._s192_stub = True
    sys.modules["opti_oignon"] = root

    projects = types.ModuleType("opti_oignon.projects")
    projects.project_store = None
    projects.ProjectFile = SimpleNamespace
    sys.modules["opti_oignon.projects"] = projects

    rag = types.ModuleType("opti_oignon.rag")
    rag.__path__ = []
    sys.modules["opti_oignon.rag"] = rag
    chunkers = types.ModuleType("opti_oignon.rag.chunkers")
    chunkers.Chunk = SimpleNamespace
    chunkers.get_chunker = None
    sys.modules["opti_oignon.rag.chunkers"] = chunkers
    rag_config = types.ModuleType("opti_oignon.rag.config")
    rag_config.get_config = lambda: SimpleNamespace(embedding=None)
    sys.modules["opti_oignon.rag.config"] = rag_config
    embeddings = types.ModuleType("opti_oignon.rag.embeddings")
    embeddings.OllamaEmbeddings = object
    sys.modules["opti_oignon.rag.embeddings"] = embeddings


def _load_standalone(basename: str):
    name = f"oo_s192_f5c_{basename}"
    if name in sys.modules:
        return sys.modules[name]
    _preseed_opti_oignon_stubs()
    spec = importlib.util.spec_from_file_location(
        name, PKG_DIR / f"{basename}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # register before exec (3.13 idiom)
    spec.loader.exec_module(module)
    return module


# =============================================================================
# Fakes
# =============================================================================

class _RecordingCollection:
    """Records add/query payloads; serves one canned retrieval result."""

    def __init__(self):
        self.added_embeddings = None
        self.queried_embeddings = None

    def count(self):
        return 1

    def get(self, where=None):
        return {"ids": []}

    def delete(self, ids=None):
        pass

    def add(self, ids, embeddings, documents, metadatas):
        self.added_embeddings = embeddings

    def query(self, query_embeddings, n_results, include):
        self.queried_embeddings = query_embeddings
        return {
            "documents": [["chunk text"]],
            "metadatas": [[{"source_file": "f.py", "chunk_index": 0}]],
            "distances": [[0.5]],
        }


def _approx(a, b, eps=1e-9):
    return all(abs(x - y) < eps for x, y in zip(a, b))


# =============================================================================
# PCX-01 -- embeddings normalized at both seams
# =============================================================================

class TestPcx01Normalization:
    def test_l2_normalize_unit(self):
        pcx = _load_standalone("project_context")
        assert _approx(pcx._l2_normalize([3.0, 4.0]), [0.6, 0.8])
        # Idempotent on already-normalized vectors (the modern-endpoint case).
        assert _approx(pcx._l2_normalize([0.6, 0.8]), [0.6, 0.8])
        # Zero vector passes through untouched.
        assert pcx._l2_normalize([0.0, 0.0]) == [0.0, 0.0]

    def test_query_embedding_normalized(self, monkeypatch):
        pcx = _load_standalone("project_context")
        monkeypatch.setattr(pcx, "CHROMADB_AVAILABLE", True)
        monkeypatch.setattr(pcx, "EMBEDDINGS_AVAILABLE", True)

        builder = object.__new__(pcx.ProjectContextBuilder)
        builder._config = {"max_chunks_per_query": 10, "min_relevance_score": 0.25}
        collection = _RecordingCollection()
        builder._get_collection = lambda pid: collection
        builder._get_embedder = lambda: SimpleNamespace(
            # Unnormalized vector: the legacy /api/embeddings shape.
            embed_single=lambda q: [3.0, 4.0],
        )

        chunks = builder.retrieve_chunks("proj1", "what is in my files")
        # Pre-fix: the raw [3, 4] went to ChromaDB; in l2 space the
        # distances blew up and every score clamped to 0 < min_score.
        assert _approx(collection.queried_embeddings[0], [0.6, 0.8])
        assert len(chunks) == 1
        assert abs(chunks[0].score - 0.75) < 1e-9  # 1 - 0.5/2

    def test_indexed_embeddings_normalized(self, monkeypatch):
        pcx = _load_standalone("project_context")
        monkeypatch.setattr(pcx, "CHROMADB_AVAILABLE", True)
        monkeypatch.setattr(pcx, "CHUNKERS_AVAILABLE", True)
        monkeypatch.setattr(pcx, "EMBEDDINGS_AVAILABLE", True)

        fake_chunk = SimpleNamespace(
            content="def foo(): pass", chunk_index=0,
            metadata={"source_file": "a.py"},
        )
        monkeypatch.setattr(
            pcx, "get_chunker",
            lambda t: SimpleNamespace(chunk=lambda c, p, ty: [fake_chunk]),
        )

        indexer = object.__new__(pcx.ProjectIndexer)
        indexer._config = {"summary_max_tokens": 200, "key_terms_max": 20}
        indexer._store = SimpleNamespace(
            get_file=lambda fid: SimpleNamespace(
                id=fid, project_id="proj1", filename="a.py",
                file_path="/tmp/a.py", file_type="code",
            ),
        )
        collection = _RecordingCollection()
        indexer._get_collection = lambda pid: collection
        indexer._get_embedder = lambda: SimpleNamespace(
            embed=lambda texts, show_progress=False: [[3.0, 4.0]],
        )
        indexer._read_file_content = lambda fp, fn: "def foo(): pass  # ok"
        indexer._update_file_record = lambda **kw: None

        result = indexer.index_file("proj1", "file1")
        assert result.success, result.error
        # Pre-fix: the raw [3, 4] was stored.
        assert _approx(collection.added_embeddings[0], [0.6, 0.8])


# =============================================================================
# PTR-01 -- L3 verdict parsing (AGL-01 class)
# =============================================================================

class TestPtr01VerdictParse:
    def _detector(self, mod, verdict: str, monkeypatch):
        monkeypatch.setattr(mod, "REQUESTS_AVAILABLE", True)

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"response": verdict}

        fake_requests = types.ModuleType("fake_requests")
        fake_requests.post = lambda *a, **kw: _Resp()
        fake_requests.exceptions = SimpleNamespace(Timeout=TimeoutError)
        monkeypatch.setattr(mod, "requests", fake_requests, raising=False)

        det = object.__new__(mod.ProjectTriggerDetector)
        det._config = dict(mod._DEFAULT_TRIGGER_CONFIG)
        det._store = SimpleNamespace(
            get_project=lambda pid: SimpleNamespace(
                name="P", description="D", settings={},
            ),
            list_files=lambda pid: [],
        )
        return det

    def test_verdict_battery(self, monkeypatch):
        mod = _load_standalone("project_triggers")
        cases = [
            ("YES", True),
            ("yes", True),                      # upper() applied
            ("NO", False),
            ("Not relevant.", False),           # NOT counts as NO
            ("NO. YES would imply files.", None),  # ambiguous (pre-fix: True)
            ("EYES", None),                     # pre-fix: substring YES -> True
            ("maybe", None),
        ]
        for verdict, expected in cases:
            det = self._detector(mod, verdict, monkeypatch)
            assert det._check_level3("query", "proj1") is expected, verdict


# =============================================================================
# RG-01 verification pins + recorded-finding pins
# =============================================================================

class TestVerificationPins:
    def test_rg01_per_project_namespacing(self):
        src = (PKG_DIR / "project_context.py").read_text(encoding="utf-8")
        # Per-project collection naming and a dedicated chroma base,
        # separate from the main rag_store chroma_v2 directory.
        assert '_COLLECTION_PREFIX = "project_"' in src
        assert 'project_chroma' in src
        rag_src = (PKG_DIR / "rag_store.py").read_text(encoding="utf-8")
        assert 'chroma_v2' in rag_src
        assert 'project_chroma' not in rag_src

    def test_prj01_no_user_scoping_pin(self):
        """projects.py has no user_id column (recorded PRJ-01, CA-04 class).

        Expected to flip when the multi-user scoping cycle lands; supersede
        with deselect + re-assert at that point.
        """
        src = (PKG_DIR / "projects.py").read_text(encoding="utf-8")
        assert "user_id" not in src

    def test_executor_skip_l3_comment_aligned(self):
        src = (PKG_DIR / "executor.py").read_text(encoding="utf-8")
        assert "skip L3 for speed" not in src
        assert "skip_l3=False" in src
