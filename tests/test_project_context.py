#!/usr/bin/env python3
"""
Tests for Project Context module (S58).

Covers:
- Helper functions (token estimation, file type mapping, key terms, summary)
- Dataclasses (IndexResult, RetrievedChunk, ProjectContext)
- ProjectIndexer (with mocked ChromaDB + embeddings)
- ProjectContextBuilder (with mocked retrieval)
- ProjectTriggerDetector (all 3 levels)
- API endpoints (4 new S58 endpoints)
- Executor integration (_inject_project_context)
- Config loading

Target: 70+ tests, 0 regressions.
"""

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import yaml

# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def tmp_dir():
    """Create a temporary directory for test data."""
    d = tempfile.mkdtemp(prefix="opti_ctx_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def config_path(tmp_dir):
    """Create a test config file with triggers + context sections."""
    config = {
        "projects": {
            "enabled": True,
            "max_projects": 5,
            "max_files_per_project": 10,
            "max_file_size_mb": 1,
            "allowed_extensions": [".txt", ".py", ".md", ".csv", ".json"],
            "default_settings": {
                "default_model": "",
                "default_pipeline": "direct",
                "context_budget_tokens": 4096,
                "auto_index": True,
            },
            "file_type_categories": {
                "text": [".txt", ".md"],
                "code": [".py", ".json"],
                "data": [".csv"],
            },
            "triggers": {
                "level1_enabled": True,
                "level2_enabled": True,
                "level3_enabled": True,
                "level2_min_matches": 2,
                "level2_min_score": 0.15,
                "level3_model": "test-model",
                "level3_timeout_ms": 500,
                "level3_ollama_url": "http://localhost:11434",
            },
            "context": {
                "summary_max_tokens": 100,
                "key_terms_max": 10,
                "max_chunks_per_query": 5,
                "min_relevance_score": 0.2,
                "context_header": "--- Test Context ---",
                "context_footer": "--- End Test Context ---",
            },
        }
    }
    p = tmp_dir / "projects.yaml"
    with open(p, "w") as f:
        yaml.dump(config, f)
    return p


@pytest.fixture
def store(tmp_dir, config_path):
    """Create a fresh ProjectStore with temp paths."""
    from opti_oignon.projects import ProjectStore
    db_path = tmp_dir / "test.db"
    storage_base = tmp_dir / "storage"
    return ProjectStore(
        db_path=db_path,
        config_path=config_path,
        storage_base=storage_base,
    )


@pytest.fixture
def project_with_files(store, tmp_dir):
    """Create a project with several text files for testing."""
    proj = store.create_project(
        name="Test Project",
        description="A project for testing context features",
        system_instructions="You are an ecology expert. Focus on biodiversity.",
        settings={"context_budget_tokens": 2048},
    )
    # Add text files
    pf1 = store.add_file(
        proj.id, "analysis.py",
        b"import pandas as pd\nimport numpy as np\n\ndef calculate_shannon_index(data):\n    \"\"\"Calculate Shannon diversity index.\"\"\"\n    proportions = data / data.sum()\n    return -sum(p * np.log(p) for p in proportions if p > 0)\n\ndef calculate_simpson_index(data):\n    return 1 - sum((n/sum(data))**2 for n in data)\n",
    )
    pf2 = store.add_file(
        proj.id, "README.md",
        b"# Biodiversity Analysis\n\nThis project analyzes species diversity using bioacoustic data from BCI Panama.\n\n## Methods\n- Shannon index\n- Simpson index\n- Metabarcoding comparison\n",
    )
    pf3 = store.add_file(
        proj.id, "data_notes.txt",
        b"Field notes from Barro Colorado Island.\nRecording stations: A1, A2, B3.\nSpecies detected: howler monkey, keel-billed toucan, red-eyed tree frog.\nWeather: tropical, 28C average.\n",
    )
    return proj, [pf1, pf2, pf3]


# =============================================================================
# TEST HELPER FUNCTIONS
# =============================================================================

class TestEstimateTokens:
    """Tests for _estimate_tokens helper."""

    def test_empty_string(self):
        from opti_oignon.project_context import _estimate_tokens
        assert _estimate_tokens("") == 0

    def test_short_text(self):
        from opti_oignon.project_context import _estimate_tokens
        # 20 chars -> ~5 tokens
        result = _estimate_tokens("Hello, how are you?!")
        assert result == 5

    def test_long_text(self):
        from opti_oignon.project_context import _estimate_tokens
        text = "a" * 4000
        assert _estimate_tokens(text) == 1000


class TestGetRagFileType:
    """Tests for _get_rag_file_type helper."""

    def test_python_extension(self):
        from opti_oignon.project_context import _get_rag_file_type
        assert _get_rag_file_type("script.py", "code") == "python"

    def test_r_extension(self):
        from opti_oignon.project_context import _get_rag_file_type
        assert _get_rag_file_type("analysis.r", "code") == "r"

    def test_markdown_extension(self):
        from opti_oignon.project_context import _get_rag_file_type
        assert _get_rag_file_type("README.md", "text") == "markdown"

    def test_csv_extension(self):
        from opti_oignon.project_context import _get_rag_file_type
        assert _get_rag_file_type("data.csv", "data") == "csv"

    def test_image_returns_none(self):
        from opti_oignon.project_context import _get_rag_file_type
        assert _get_rag_file_type("photo.png", "image") is None

    def test_archive_returns_none(self):
        from opti_oignon.project_context import _get_rag_file_type
        assert _get_rag_file_type("backup.zip", "archive") is None

    def test_unknown_ext_falls_back_to_category(self):
        from opti_oignon.project_context import _get_rag_file_type
        assert _get_rag_file_type("weird.xyz", "text") == "text"

    def test_shell_extension(self):
        from opti_oignon.project_context import _get_rag_file_type
        assert _get_rag_file_type("run.sh", "code") == "shell"

    def test_javascript_extension(self):
        from opti_oignon.project_context import _get_rag_file_type
        assert _get_rag_file_type("app.js", "code") == "javascript"


class TestExtractKeyTerms:
    """Tests for _extract_key_terms helper."""

    def test_empty_text(self):
        from opti_oignon.project_context import _extract_key_terms
        assert _extract_key_terms("") == []

    def test_extracts_meaningful_terms(self):
        from opti_oignon.project_context import _extract_key_terms
        text = "Shannon index diversity species biodiversity analysis data"
        terms = _extract_key_terms(text, max_terms=5)
        assert len(terms) <= 5
        assert all(isinstance(t, str) for t in terms)

    def test_filters_stop_words(self):
        from opti_oignon.project_context import _extract_key_terms
        text = "the and is are for with but not this that"
        terms = _extract_key_terms(text, max_terms=10)
        assert len(terms) == 0

    def test_filters_short_words(self):
        from opti_oignon.project_context import _extract_key_terms
        text = "a to it is on at by"
        terms = _extract_key_terms(text, max_terms=10)
        assert len(terms) == 0

    def test_respects_max_terms(self):
        from opti_oignon.project_context import _extract_key_terms
        text = " ".join(f"term{i}" for i in range(100))
        terms = _extract_key_terms(text, max_terms=5)
        assert len(terms) == 5

    def test_returns_lowercase(self):
        from opti_oignon.project_context import _extract_key_terms
        text = "Shannon DIVERSITY Biodiversity"
        terms = _extract_key_terms(text, max_terms=10)
        for t in terms:
            assert t == t.lower()

    def test_frequency_ordering(self):
        from opti_oignon.project_context import _extract_key_terms
        text = "alpha beta alpha gamma alpha beta delta"
        terms = _extract_key_terms(text, max_terms=10)
        # alpha appears 3x, should be first
        assert terms[0] == "alpha"


class TestGenerateSummary:
    """Tests for _generate_summary helper."""

    def test_short_text_unchanged(self):
        from opti_oignon.project_context import _generate_summary
        text = "Short text."
        assert _generate_summary(text, max_tokens=100) == "Short text."

    def test_truncates_long_text(self):
        from opti_oignon.project_context import _generate_summary
        text = "A" * 10000
        result = _generate_summary(text, max_tokens=50)
        assert len(result) < len(text)

    def test_breaks_at_sentence(self):
        from opti_oignon.project_context import _generate_summary
        text = "First sentence. Second sentence. Third sentence. " + "X" * 5000
        result = _generate_summary(text, max_tokens=20)
        # Should end near a period
        assert "." in result

    def test_empty_text(self):
        from opti_oignon.project_context import _generate_summary
        assert _generate_summary("", max_tokens=100) == ""


# =============================================================================
# TEST DATACLASSES
# =============================================================================

class TestIndexResult:
    """Tests for IndexResult dataclass."""

    def test_defaults(self):
        from opti_oignon.project_context import IndexResult
        r = IndexResult()
        assert r.file_id == ""
        assert r.success is False
        assert r.chunk_count == 0
        assert r.key_terms == []
        assert r.error == ""

    def test_with_values(self):
        from opti_oignon.project_context import IndexResult
        r = IndexResult(
            file_id="f1", filename="test.py",
            success=True, chunk_count=5,
            summary="A test file", key_terms=["test", "python"],
        )
        assert r.success is True
        assert r.chunk_count == 5
        assert len(r.key_terms) == 2


class TestRetrievedChunk:
    """Tests for RetrievedChunk dataclass."""

    def test_defaults(self):
        from opti_oignon.project_context import RetrievedChunk
        c = RetrievedChunk()
        assert c.content == ""
        assert c.score == 0.0
        assert c.metadata == {}

    def test_with_values(self):
        from opti_oignon.project_context import RetrievedChunk
        c = RetrievedChunk(content="hello", score=0.85, source_file="test.py")
        assert c.score == 0.85


class TestProjectContextDataclass:
    """Tests for ProjectContext dataclass."""

    def test_defaults(self):
        from opti_oignon.project_context import ProjectContext
        ctx = ProjectContext()
        assert ctx.context_text == ""
        assert ctx.chunks_used == 0
        assert ctx.source_files == []

    def test_with_values(self):
        from opti_oignon.project_context import ProjectContext
        ctx = ProjectContext(
            context_text="some context",
            system_instructions="be helpful",
            chunks_used=3,
            total_tokens_estimate=100,
            source_files=["a.py", "b.md"],
            project_id="p1",
            project_name="Test",
        )
        assert ctx.chunks_used == 3
        assert len(ctx.source_files) == 2


# =============================================================================
# TEST PROJECT INDEXER
# =============================================================================

class TestProjectIndexer:
    """Tests for ProjectIndexer (with mocked dependencies)."""

    def test_available_when_deps_present(self):
        from opti_oignon.project_context import ProjectIndexer
        indexer = ProjectIndexer()
        # In test env, chromadb may or may not be installed
        # Just verify the property works
        assert isinstance(indexer.available, bool)

    def test_index_file_missing_deps(self, store, project_with_files):
        """Indexation returns error when ChromaDB is missing."""
        from opti_oignon.project_context import ProjectIndexer
        proj, files = project_with_files
        indexer = ProjectIndexer(store=store)

        with patch("opti_oignon.project_context.CHROMADB_AVAILABLE", False):
            result = indexer.index_file(proj.id, files[0].id)
            assert result.success is False
            assert "Missing dependencies" in result.error

    def test_index_file_not_found(self, store, tmp_dir):
        """Indexation returns error for nonexistent file."""
        from opti_oignon.project_context import ProjectIndexer
        proj = store.create_project(name="Empty")
        indexer = ProjectIndexer(store=store, chroma_base=tmp_dir / "chroma")
        with patch("opti_oignon.project_context.CHROMADB_AVAILABLE", True), \
             patch("opti_oignon.project_context.CHUNKERS_AVAILABLE", True), \
             patch("opti_oignon.project_context.EMBEDDINGS_AVAILABLE", True):
            result = indexer.index_file(proj.id, "nonexistent-id")
        assert result.success is False
        assert "not found" in result.error

    def test_index_file_binary_type(self, store, tmp_dir):
        """Indexation skips binary file types (images)."""
        from opti_oignon.project_context import ProjectIndexer
        proj = store.create_project(name="Binary")
        # Need to add a .png to the allowed extensions for this store
        pf = store.add_file(proj.id, "photo.txt", b"not actually an image")
        indexer = ProjectIndexer(store=store, chroma_base=tmp_dir / "chroma")

        # Mock the file type to be image
        with patch.object(store, "get_file") as mock_get, \
             patch("opti_oignon.project_context.CHROMADB_AVAILABLE", True), \
             patch("opti_oignon.project_context.CHUNKERS_AVAILABLE", True), \
             patch("opti_oignon.project_context.EMBEDDINGS_AVAILABLE", True):
            mock_file = MagicMock()
            mock_file.filename = "photo.png"
            mock_file.file_type = "image"
            mock_file.file_path = pf.file_path
            mock_get.return_value = mock_file
            result = indexer.index_file(proj.id, pf.id)
            assert result.success is False
            assert "not indexable" in result.error

    def test_index_file_success_mocked(self, store, project_with_files, tmp_dir):
        """Full indexation with mocked ChromaDB and embedder."""
        from opti_oignon.project_context import ProjectIndexer
        proj, files = project_with_files
        chroma_base = tmp_dir / "chroma"
        indexer = ProjectIndexer(store=store, chroma_base=chroma_base)

        # Mock the collection
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": []}
        mock_collection.add.return_value = None

        # Mock embedder that returns fake vectors
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [[0.1] * 1024 for _ in range(10)]

        # Mock chunker that returns fake chunks
        mock_chunk = MagicMock()
        mock_chunk.content = "def calculate_shannon_index(data): pass"
        mock_chunk.chunk_index = 0
        mock_chunk.metadata = {"source_file": files[0].file_path, "chunk_index": 0}
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = [mock_chunk]

        with patch.object(indexer, "_get_collection", return_value=mock_collection), \
             patch.object(indexer, "_get_embedder", return_value=mock_embedder), \
             patch("opti_oignon.project_context.CHROMADB_AVAILABLE", True), \
             patch("opti_oignon.project_context.CHUNKERS_AVAILABLE", True), \
             patch("opti_oignon.project_context.EMBEDDINGS_AVAILABLE", True), \
             patch("opti_oignon.project_context.get_chunker", return_value=mock_chunker):
            result = indexer.index_file(proj.id, files[0].id)

        assert result.success is True
        assert result.chunk_count > 0
        assert result.filename == "analysis.py"
        assert len(result.key_terms) > 0
        assert len(result.summary) > 0
        # Verify ChromaDB was called
        mock_collection.add.assert_called_once()

    def test_index_file_updates_sqlite(self, store, project_with_files, tmp_dir):
        """Indexation updates the file record in SQLite."""
        from opti_oignon.project_context import ProjectIndexer
        proj, files = project_with_files
        chroma_base = tmp_dir / "chroma"
        indexer = ProjectIndexer(store=store, chroma_base=chroma_base)

        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": []}
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [[0.1] * 1024 for _ in range(10)]

        mock_chunk = MagicMock()
        mock_chunk.content = "import pandas"
        mock_chunk.chunk_index = 0
        mock_chunk.metadata = {"source_file": files[0].file_path, "chunk_index": 0}
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = [mock_chunk]

        with patch.object(indexer, "_get_collection", return_value=mock_collection), \
             patch.object(indexer, "_get_embedder", return_value=mock_embedder), \
             patch("opti_oignon.project_context.CHROMADB_AVAILABLE", True), \
             patch("opti_oignon.project_context.CHUNKERS_AVAILABLE", True), \
             patch("opti_oignon.project_context.EMBEDDINGS_AVAILABLE", True), \
             patch("opti_oignon.project_context.get_chunker", return_value=mock_chunker):
            indexer.index_file(proj.id, files[0].id)

        # Check SQLite was updated
        updated_file = store.get_file(files[0].id)
        assert updated_file.indexed is True
        assert updated_file.chunk_count > 0
        assert len(updated_file.summary) > 0
        assert len(updated_file.key_terms) > 0

    def test_remove_file_from_index(self, store, project_with_files, tmp_dir):
        """Remove file chunks from ChromaDB index."""
        from opti_oignon.project_context import ProjectIndexer
        proj, files = project_with_files
        indexer = ProjectIndexer(store=store, chroma_base=tmp_dir / "chroma")

        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": ["chunk1", "chunk2"]}

        with patch.object(indexer, "_get_collection", return_value=mock_collection), \
             patch("opti_oignon.project_context.CHROMADB_AVAILABLE", True):
            result = indexer.remove_file_from_index(proj.id, files[0].id)

        assert result is True
        mock_collection.delete.assert_called_once_with(ids=["chunk1", "chunk2"])

    def test_reindex_project(self, store, project_with_files, tmp_dir):
        """Reindex processes all project files."""
        from opti_oignon.project_context import ProjectIndexer
        proj, files = project_with_files
        indexer = ProjectIndexer(store=store, chroma_base=tmp_dir / "chroma")

        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": []}
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [[0.1] * 1024 for _ in range(10)]

        mock_chunk = MagicMock()
        mock_chunk.content = "some content"
        mock_chunk.chunk_index = 0
        mock_chunk.metadata = {"source_file": "test", "chunk_index": 0}
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = [mock_chunk]

        with patch.object(indexer, "_get_collection", return_value=mock_collection), \
             patch.object(indexer, "_get_embedder", return_value=mock_embedder), \
             patch("opti_oignon.project_context.CHROMADB_AVAILABLE", True), \
             patch("opti_oignon.project_context.CHUNKERS_AVAILABLE", True), \
             patch("opti_oignon.project_context.EMBEDDINGS_AVAILABLE", True), \
             patch("opti_oignon.project_context.get_chunker", return_value=mock_chunker):
            results = indexer.reindex_project(proj.id)

        assert len(results) == 3
        succeeded = [r for r in results if r.success]
        assert len(succeeded) == 3

    def test_read_file_content_text(self, store, project_with_files, tmp_dir):
        """Reading a text file returns its content."""
        from opti_oignon.project_context import ProjectIndexer
        proj, files = project_with_files
        indexer = ProjectIndexer(store=store, chroma_base=tmp_dir / "chroma")
        content = indexer._read_file_content(files[0].file_path, files[0].filename)
        assert content is not None
        assert "shannon" in content.lower()

    def test_read_file_content_nonexistent(self, tmp_dir):
        """Reading a nonexistent file returns None."""
        from opti_oignon.project_context import ProjectIndexer
        indexer = ProjectIndexer(chroma_base=tmp_dir / "chroma")
        content = indexer._read_file_content("/nonexistent/file.py", "file.py")
        assert content is None

    def test_delete_project_index_no_chromadb(self, tmp_dir):
        """Delete project index gracefully handles missing ChromaDB."""
        from opti_oignon.project_context import ProjectIndexer
        indexer = ProjectIndexer(chroma_base=tmp_dir / "chroma")
        with patch("opti_oignon.project_context.CHROMADB_AVAILABLE", False):
            result = indexer.delete_project_index("some-id")
            assert result is False


# =============================================================================
# TEST PROJECT CONTEXT BUILDER
# =============================================================================

class TestProjectContextBuilder:
    """Tests for ProjectContextBuilder."""

    def test_build_system_instructions_only(self, store, project_with_files, tmp_dir):
        """build_system_instructions_only returns project instructions."""
        from opti_oignon.project_context import ProjectContextBuilder
        proj, _ = project_with_files
        builder = ProjectContextBuilder(store=store, chroma_base=tmp_dir / "chroma")
        ctx = builder.build_system_instructions_only(proj.id)
        assert "ecology expert" in ctx.system_instructions.lower()
        assert ctx.context_text == ctx.system_instructions
        assert ctx.chunks_used == 0

    def test_build_system_instructions_empty(self, store, tmp_dir):
        """Empty system_instructions project returns empty context."""
        from opti_oignon.project_context import ProjectContextBuilder
        proj = store.create_project(name="NoInstructions")
        builder = ProjectContextBuilder(store=store, chroma_base=tmp_dir / "chroma")
        ctx = builder.build_system_instructions_only(proj.id)
        assert ctx.system_instructions == ""
        assert ctx.context_text == ""

    def test_build_system_instructions_nonexistent_project(self, store, tmp_dir):
        """Nonexistent project returns empty context."""
        from opti_oignon.project_context import ProjectContextBuilder
        builder = ProjectContextBuilder(store=store, chroma_base=tmp_dir / "chroma")
        ctx = builder.build_system_instructions_only("nonexistent")
        assert ctx.context_text == ""

    def test_retrieve_chunks_empty_query(self, store, tmp_dir):
        """Empty query returns no chunks."""
        from opti_oignon.project_context import ProjectContextBuilder
        builder = ProjectContextBuilder(store=store, chroma_base=tmp_dir / "chroma")
        chunks = builder.retrieve_chunks("some-project", "")
        assert chunks == []

    def test_retrieve_chunks_no_chromadb(self, store, tmp_dir):
        """Retrieval returns empty when ChromaDB unavailable."""
        from opti_oignon.project_context import ProjectContextBuilder
        builder = ProjectContextBuilder(store=store, chroma_base=tmp_dir / "chroma")
        with patch("opti_oignon.project_context.CHROMADB_AVAILABLE", False):
            chunks = builder.retrieve_chunks("some-project", "test query")
            assert chunks == []

    def test_build_context_with_mocked_retrieval(self, store, project_with_files, tmp_dir):
        """Build context assembles chunks within budget."""
        from opti_oignon.project_context import ProjectContextBuilder, RetrievedChunk
        proj, _ = project_with_files
        builder = ProjectContextBuilder(store=store, chroma_base=tmp_dir / "chroma")

        fake_chunks = [
            RetrievedChunk(content="Shannon index measures diversity.", score=0.9, source_file="/test/analysis.py"),
            RetrievedChunk(content="Species data from BCI Panama.", score=0.8, source_file="/test/data.txt"),
        ]
        with patch.object(builder, "retrieve_chunks", return_value=fake_chunks):
            ctx = builder.build_context(proj.id, "Shannon diversity", budget_tokens=2048)

        assert ctx.chunks_used == 2
        assert "ecology expert" in ctx.context_text.lower()
        assert "Shannon" in ctx.context_text
        assert len(ctx.source_files) == 2

    def test_build_context_respects_budget(self, store, project_with_files, tmp_dir):
        """Context builder never exceeds token budget."""
        from opti_oignon.project_context import (
            ProjectContextBuilder,
            RetrievedChunk,
            _estimate_tokens,
        )
        proj, _ = project_with_files
        builder = ProjectContextBuilder(store=store, chroma_base=tmp_dir / "chroma")

        # Create chunks that together exceed budget
        big_chunk = RetrievedChunk(content="X" * 10000, score=0.9, source_file="big.py")
        small_chunk = RetrievedChunk(content="Small data.", score=0.8, source_file="small.py")

        with patch.object(builder, "retrieve_chunks", return_value=[big_chunk, small_chunk]):
            ctx = builder.build_context(proj.id, "test", budget_tokens=500)

        # Total tokens should be within budget (roughly)
        assert ctx.total_tokens_estimate <= 600  # Allow some overhead

    def test_build_context_no_store(self, tmp_dir):
        """Build context without store returns empty."""
        from opti_oignon.project_context import ProjectContextBuilder
        builder = ProjectContextBuilder(store=None, chroma_base=tmp_dir / "chroma")
        ctx = builder.build_context("any-id", "query")
        assert ctx.context_text == ""


# =============================================================================
# TEST TRIGGER DETECTOR - LEVEL 1 (REGEX)
# =============================================================================

class TestTriggerLevel1:
    """Tests for Level 1 regex trigger detection."""

    def _detect(self, query, store=None, project_id="p1"):
        from opti_oignon.project_triggers import ProjectTriggerDetector
        detector = ProjectTriggerDetector(store=store)
        return detector.detect(query, project_id, skip_l3=True)

    def test_at_project_trigger(self):
        result = self._detect("@project what files do I have?")
        assert result.relevant is True
        assert result.trigger_level == 1

    def test_project_files_trigger(self):
        result = self._detect("look in the project files for diversity data")
        assert result.relevant is True
        assert result.trigger_level == 1

    def test_look_in_files_trigger(self):
        result = self._detect("look in the files for Shannon index")
        assert result.relevant is True
        assert result.trigger_level == 1

    def test_search_in_project_trigger(self):
        result = self._detect("search in the project for species data")
        assert result.relevant is True
        assert result.trigger_level == 1

    def test_uploaded_files_trigger(self):
        result = self._detect("check my uploaded files")
        assert result.relevant is True
        assert result.trigger_level == 1

    def test_french_dans_le_projet(self):
        result = self._detect("cherche dans le projet les donnees acoustiques")
        assert result.relevant is True
        assert result.trigger_level == 1

    def test_french_mes_fichiers(self):
        result = self._detect("regarde dans mes fichiers")
        assert result.relevant is True
        assert result.trigger_level == 1

    def test_no_trigger_general_question(self):
        result = self._detect("What is the capital of France?")
        assert result.trigger_level != 1

    def test_no_trigger_code_question(self):
        result = self._detect("How do I sort a list in Python?")
        assert result.trigger_level != 1

    def test_confidence_high(self):
        result = self._detect("@project show summary")
        assert result.confidence >= 0.9

    def test_case_insensitive(self):
        result = self._detect("LOOK IN THE PROJECT FILES")
        assert result.relevant is True
        assert result.trigger_level == 1

    def test_empty_query(self):
        result = self._detect("")
        assert result.relevant is False
        assert result.trigger_level == 0


# =============================================================================
# TEST TRIGGER DETECTOR - LEVEL 2 (TERM MATCHING)
# =============================================================================

class TestTriggerLevel2:
    """Tests for Level 2 term matching trigger detection."""

    def _make_store_with_terms(self, terms):
        """Create a mock store that returns files with given key_terms."""
        mock_store = MagicMock()
        mock_file = MagicMock()
        mock_file.indexed = True
        mock_file.key_terms = terms
        mock_store.list_files.return_value = [mock_file]
        mock_store.get_project.return_value = MagicMock(
            name="Test", description="", settings={}
        )
        return mock_store

    def test_matches_key_terms(self):
        from opti_oignon.project_triggers import ProjectTriggerDetector
        store = self._make_store_with_terms(["shannon", "diversity", "species", "index"])
        detector = ProjectTriggerDetector(store=store)
        # Query with 2+ matching terms
        result = detector.detect("calculate shannon diversity", "p1", skip_l3=True)
        assert result.relevant is True
        assert result.trigger_level == 2
        assert "shannon" in result.matched_terms
        assert "diversity" in result.matched_terms

    def test_no_match_unrelated_query(self):
        from opti_oignon.project_triggers import ProjectTriggerDetector
        store = self._make_store_with_terms(["shannon", "diversity", "species"])
        detector = ProjectTriggerDetector(store=store)
        result = detector.detect("weather forecast tomorrow paris", "p1", skip_l3=True)
        assert result.trigger_level != 2

    def test_requires_min_matches(self):
        from opti_oignon.project_triggers import ProjectTriggerDetector
        store = self._make_store_with_terms(["shannon", "diversity", "species"])
        detector = ProjectTriggerDetector(store=store)
        # Only 1 match - below default min of 2
        result = detector.detect("shannon equation math", "p1", skip_l3=True)
        # May or may not trigger depending on exact min_matches config
        if result.trigger_level == 2:
            assert len(result.matched_terms) >= 2

    def test_no_indexed_files(self):
        from opti_oignon.project_triggers import ProjectTriggerDetector
        mock_store = MagicMock()
        mock_file = MagicMock()
        mock_file.indexed = False
        mock_file.key_terms = []
        mock_store.list_files.return_value = [mock_file]
        detector = ProjectTriggerDetector(store=mock_store)
        result = detector.detect("shannon diversity", "p1", skip_l3=True)
        assert result.trigger_level != 2

    def test_empty_key_terms(self):
        from opti_oignon.project_triggers import ProjectTriggerDetector
        store = self._make_store_with_terms([])
        detector = ProjectTriggerDetector(store=store)
        result = detector.detect("any query here", "p1", skip_l3=True)
        assert result.trigger_level != 2

    def test_score_calculation(self):
        from opti_oignon.project_triggers import ProjectTriggerDetector
        store = self._make_store_with_terms(["alpha", "beta", "gamma", "delta"])
        detector = ProjectTriggerDetector(store=store)
        # All 4 query tokens match
        result = detector.detect("alpha beta gamma delta", "p1", skip_l3=True)
        assert result.relevant is True
        assert result.confidence > 0.7


# =============================================================================
# TEST TRIGGER DETECTOR - LEVEL 3 (LLM)
# =============================================================================

class TestTriggerLevel3:
    """Tests for Level 3 LLM classification trigger detection."""

    def _make_store(self):
        mock_store = MagicMock()
        mock_project = MagicMock()
        mock_project.name = "BCI Acoustics"
        mock_project.description = "Bioacoustic analysis"
        mock_project.settings = {"default_model": "test-model"}
        mock_store.get_project.return_value = mock_project
        mock_file = MagicMock()
        mock_file.filename = "data.csv"
        mock_file.indexed = False
        mock_file.key_terms = []
        mock_store.list_files.return_value = [mock_file]
        return mock_store

    def test_llm_says_yes(self):
        from opti_oignon.project_triggers import ProjectTriggerDetector
        store = self._make_store()
        detector = ProjectTriggerDetector(store=store)

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "YES"}
        mock_response.raise_for_status.return_value = None

        with patch("opti_oignon.project_triggers.requests.post", return_value=mock_response), \
             patch("opti_oignon.project_triggers.REQUESTS_AVAILABLE", True):
            result = detector._check_level3("analyze species diversity", "p1")
        assert result is True

    def test_llm_says_no(self):
        from opti_oignon.project_triggers import ProjectTriggerDetector
        store = self._make_store()
        detector = ProjectTriggerDetector(store=store)

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "NO"}
        mock_response.raise_for_status.return_value = None

        with patch("opti_oignon.project_triggers.requests.post", return_value=mock_response), \
             patch("opti_oignon.project_triggers.REQUESTS_AVAILABLE", True):
            result = detector._check_level3("what is the weather", "p1")
        assert result is False

    def test_llm_timeout(self):
        import requests

        from opti_oignon.project_triggers import ProjectTriggerDetector
        store = self._make_store()
        detector = ProjectTriggerDetector(store=store)

        with patch("opti_oignon.project_triggers.requests.post",
                    side_effect=requests.exceptions.Timeout), \
             patch("opti_oignon.project_triggers.REQUESTS_AVAILABLE", True):
            result = detector._check_level3("test query", "p1")
        assert result is None

    def test_llm_ambiguous_response(self):
        from opti_oignon.project_triggers import ProjectTriggerDetector
        store = self._make_store()
        detector = ProjectTriggerDetector(store=store)

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "MAYBE"}
        mock_response.raise_for_status.return_value = None

        with patch("opti_oignon.project_triggers.requests.post", return_value=mock_response), \
             patch("opti_oignon.project_triggers.REQUESTS_AVAILABLE", True):
            result = detector._check_level3("ambiguous query", "p1")
        assert result is None

    def test_no_store(self):
        from opti_oignon.project_triggers import ProjectTriggerDetector
        detector = ProjectTriggerDetector(store=None)
        result = detector._check_level3("query", "p1")
        assert result is None


# =============================================================================
# TEST TRIGGER DETECTOR - ESCALATION
# =============================================================================

class TestTriggerEscalation:
    """Tests for trigger level escalation behavior."""

    def test_l1_stops_before_l2(self):
        from opti_oignon.project_triggers import ProjectTriggerDetector
        mock_store = MagicMock()
        detector = ProjectTriggerDetector(store=mock_store)
        result = detector.detect("@project show data", "p1", skip_l3=True)
        assert result.trigger_level == 1
        # L2 was never called
        mock_store.list_files.assert_not_called()

    def test_l2_runs_when_l1_fails(self):
        from opti_oignon.project_triggers import ProjectTriggerDetector
        mock_store = MagicMock()
        mock_file = MagicMock()
        mock_file.indexed = True
        mock_file.key_terms = ["diversity", "species", "shannon"]
        mock_store.list_files.return_value = [mock_file]
        detector = ProjectTriggerDetector(store=mock_store)
        result = detector.detect("calculate diversity species", "p1", skip_l3=True)
        # Should hit L2
        if result.relevant:
            assert result.trigger_level == 2

    def test_duration_recorded(self):
        from opti_oignon.project_triggers import ProjectTriggerDetector
        detector = ProjectTriggerDetector(store=MagicMock())
        result = detector.detect("@project test", "p1", skip_l3=True)
        assert result.duration_ms >= 0

    def test_skip_l3_respected(self):
        from opti_oignon.project_triggers import ProjectTriggerDetector
        mock_store = MagicMock()
        mock_file = MagicMock()
        mock_file.indexed = False
        mock_file.key_terms = []
        mock_store.list_files.return_value = [mock_file]
        detector = ProjectTriggerDetector(store=mock_store)

        with patch.object(detector, "_check_level3") as mock_l3:
            detector.detect("general question", "p1", skip_l3=True)
            mock_l3.assert_not_called()


# =============================================================================
# TEST RELEVANCE RESULT DATACLASS
# =============================================================================

class TestRelevanceResult:
    """Tests for RelevanceResult dataclass."""

    def test_defaults(self):
        from opti_oignon.project_triggers import RelevanceResult
        r = RelevanceResult()
        assert r.relevant is False
        assert r.confidence == 0.0
        assert r.trigger_level == 0
        assert r.matched_terms == []

    def test_with_values(self):
        from opti_oignon.project_triggers import RelevanceResult
        r = RelevanceResult(
            relevant=True, confidence=0.85,
            trigger_level=2, matched_terms=["alpha", "beta"],
        )
        assert r.relevant is True
        assert r.confidence == 0.85


# =============================================================================
# TEST CONFIG LOADING
# =============================================================================

class TestConfigLoading:
    """Tests for config loading in both modules."""

    def test_trigger_config_defaults(self):
        from opti_oignon.project_triggers import _load_trigger_config
        # With a missing config file, returns defaults
        with patch("opti_oignon.project_triggers._CONFIG_DIR", Path("/nonexistent")):
            config = _load_trigger_config()
        assert config["level1_enabled"] is True
        assert config["level2_min_matches"] == 2
        assert config["level3_timeout_ms"] == 500

    def test_context_config_defaults(self):
        from opti_oignon.project_context import _load_context_config
        with patch("opti_oignon.project_context._CONFIG_DIR", Path("/nonexistent")):
            config = _load_context_config()
        assert config["summary_max_tokens"] == 200
        assert config["max_chunks_per_query"] == 10
        assert config["min_relevance_score"] == 0.25


# =============================================================================
# TEST EXECUTOR INTEGRATION
# =============================================================================

class TestExecutorIntegration:
    """Tests for _inject_project_context in executor.py."""

    def test_no_conversation_id_returns_unchanged(self):
        """Without conversation_id, system prompt is unchanged."""
        from opti_oignon.executor import Executor
        ex = Executor()
        result = ex._inject_project_context("original prompt", "question", None)
        assert result == "original prompt"

    def test_no_linked_project_returns_unchanged(self):
        """When conversation has no linked project, prompt unchanged."""
        from opti_oignon.executor import Executor
        ex = Executor()
        with patch("opti_oignon.executor.PROJECT_CONTEXT_AVAILABLE", True), \
             patch("opti_oignon.executor._project_store") as mock_store:
            mock_store.get_project_for_conversation.return_value = None
            result = ex._inject_project_context("original", "question", "conv-123")
        assert result == "original"

    def test_injects_system_instructions(self):
        """Project system_instructions are injected."""
        from opti_oignon.executor import Executor
        from opti_oignon.project_context import ProjectContext
        ex = Executor()

        mock_ctx = ProjectContext(
            context_text="You are an ecology expert.",
            system_instructions="You are an ecology expert.",
            chunks_used=0,
            total_tokens_estimate=10,
        )

        with patch("opti_oignon.executor.PROJECT_CONTEXT_AVAILABLE", True), \
             patch("opti_oignon.executor._project_store") as mock_store, \
             patch("opti_oignon.executor._trigger_detector") as mock_trigger, \
             patch("opti_oignon.executor._project_context_builder") as mock_builder:
            mock_store.get_project_for_conversation.return_value = "proj-1"
            mock_trigger.detect.return_value = MagicMock(relevant=False)
            mock_builder.available = False
            mock_builder.build_system_instructions_only.return_value = mock_ctx
            result = ex._inject_project_context("base prompt", "hello", "conv-1")

        assert "ecology expert" in result

    def test_project_context_unavailable_passthrough(self):
        """When PROJECT_CONTEXT_AVAILABLE is False, prompt unchanged."""
        from opti_oignon.executor import Executor
        ex = Executor()
        with patch("opti_oignon.executor.PROJECT_CONTEXT_AVAILABLE", False):
            result = ex._inject_project_context("original", "q", "conv-1")
        assert result == "original"


# =============================================================================
# TEST API ENDPOINTS (S58)
# =============================================================================

class TestProjectContextAPI:
    """Tests for the 4 new S58 API endpoints."""

    @classmethod
    def setup_class(cls):
        """Set up the TestClient."""
        from fastapi.testclient import TestClient

        from opti_oignon.api.app import app
        cls.client = TestClient(app)

    def _create_project_with_file(self):
        """Helper: create a project and upload a file."""
        cr = self.client.post("/api/projects", json={
            "name": "CtxTest",
            "system_instructions": "Be helpful.",
        })
        pid = cr.json()["id"]
        fr = self.client.post(
            f"/api/projects/{pid}/files",
            files={"file": ("test.py", b"def hello():\n    return 'world'\n", "text/plain")},
        )
        fid = fr.json()["id"]
        return pid, fid

    def test_index_file_endpoint_not_found(self):
        """Index file returns 404 for nonexistent file."""
        cr = self.client.post("/api/projects", json={"name": "IdxTest"})
        pid = cr.json()["id"]
        r = self.client.post(f"/api/projects/{pid}/files/nonexistent/index")
        assert r.status_code == 404

    def test_reindex_endpoint_not_found(self):
        """Reindex returns 404 for nonexistent project."""
        r = self.client.post("/api/projects/nonexistent/reindex")
        assert r.status_code == 404

    def test_context_preview_endpoint(self):
        """Context preview returns structured response."""
        cr = self.client.post("/api/projects", json={
            "name": "PreviewTest",
            "system_instructions": "Test instructions.",
        })
        pid = cr.json()["id"]
        r = self.client.get(f"/api/projects/{pid}/context", params={"query": "hello world"})
        assert r.status_code == 200
        data = r.json()
        assert data["project_id"] == pid
        assert data["query"] == "hello world"
        assert "trigger" in data
        assert "context" in data

    def test_context_preview_missing_query(self):
        """Context preview returns 422 without query param."""
        cr = self.client.post("/api/projects", json={"name": "NoQuery"})
        pid = cr.json()["id"]
        r = self.client.get(f"/api/projects/{pid}/context")
        assert r.status_code == 422

    def test_context_preview_not_found(self):
        """Context preview returns 404 for nonexistent project."""
        r = self.client.get("/api/projects/nonexistent/context", params={"query": "test"})
        assert r.status_code == 404

    def test_file_summary_endpoint(self):
        """File summary returns indexed info."""
        pid, fid = self._create_project_with_file()
        r = self.client.get(f"/api/projects/{pid}/files/{fid}/summary")
        assert r.status_code == 200
        data = r.json()
        assert data["file_id"] == fid
        assert data["filename"] == "test.py"
        assert "indexed" in data
        assert "summary" in data
        assert "key_terms" in data

    def test_file_summary_not_found(self):
        """File summary returns 404 for nonexistent file."""
        cr = self.client.post("/api/projects", json={"name": "SumTest"})
        pid = cr.json()["id"]
        r = self.client.get(f"/api/projects/{pid}/files/nonexistent/summary")
        assert r.status_code == 404

    def test_health_includes_s58_modules(self):
        """Health check includes project_context and project_triggers."""
        r = self.client.get("/api/health")
        data = r.json()
        modules = data["modules"]
        assert "project_context" in modules
        assert "project_triggers" in modules


# =============================================================================
# TEST TOKENIZE QUERY HELPER
# =============================================================================

class TestTokenizeQuery:
    """Tests for _tokenize_query helper."""

    def test_basic_tokenization(self):
        from opti_oignon.project_triggers import _tokenize_query
        tokens = _tokenize_query("Hello world test")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_filters_short_tokens(self):
        from opti_oignon.project_triggers import _tokenize_query
        tokens = _tokenize_query("a to it ok yes no")
        # All are < 3 chars, should be empty
        assert "a" not in tokens
        assert "to" not in tokens

    def test_empty_string(self):
        from opti_oignon.project_triggers import _tokenize_query
        tokens = _tokenize_query("")
        assert len(tokens) == 0

    def test_returns_lowercase(self):
        from opti_oignon.project_triggers import _tokenize_query
        tokens = _tokenize_query("Shannon INDEX")
        assert "shannon" in tokens
        assert "index" in tokens
