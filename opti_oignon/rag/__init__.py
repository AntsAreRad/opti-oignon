#!/usr/bin/env python3
"""
RAG MODULE - Opti-Oignon Retrieval-Augmented Generation
=======================================================

Personal RAG system to enrich queries with context from your documents.

Modules:
- config: Centralized configuration
- chunkers: Smart document chunking
- embeddings: Ollama interface for embeddings
- indexer: Document indexing into ChromaDB
- retriever: Semantic search
- augmenter: Augmented prompt generation
- batch_ingest: Batch ingestion engine with background processing (S119)

Quick usage:
    from opti_oignon.rag import ContexteurRAGIntegration

    rag = ContexteurRAGIntegration()
    rag.index_folder("~/Documents/code")
    results = rag.search("diversity index")
    enriched = rag.enrich_query("How to calculate Shannon index?")

CLI:
    python -m opti_oignon.rag index ~/Documents/code
    python -m opti_oignon.rag search "Shannon index"
    python -m opti_oignon.rag stats

Author: Léon
"""

__version__ = "2.0.1"
__author__ = "Léon"

# Main imports
from .augmenter import (
    AugmentedPrompt,
    ContexteurRAGIntegration,
    PromptAugmenter,
    quick_augment,
)
from .chunkers import (
    BaseChunker,
    Chunk,
    CodeChunker,
    CSVChunker,
    MarkdownChunker,
    RChunker,
    TextChunker,
    get_chunker,
)
from .config import (
    ChunkingConfig,
    EmbeddingConfig,
    RAGConfig,
    RetrieverConfig,
    get_config,
    set_config,
)
from .embeddings import (
    CachedEmbeddings,
    OllamaEmbeddings,
    check_ollama_status,
    normalize_embeddings,
)
from .batch_ingest import (
    BatchIngestEngine,
    FileStatus,
    IngestFileRecord,
    IngestJobRecord,
    JobStatus,
    get_batch_ingest_engine,
    scan_folder,
)
from .indexer import (
    DocumentIndexer,
    quick_index,
)
from .retriever import (
    DocumentRetriever,
    SearchResult,
    format_results,
    quick_search,
)

# Public exports
__all__ = [
    # Version
    "__version__",

    # Config
    "get_config",
    "set_config",
    "RAGConfig",
    "ChunkingConfig",
    "EmbeddingConfig",
    "RetrieverConfig",

    # Chunkers
    "Chunk",
    "get_chunker",
    "BaseChunker",
    "CodeChunker",
    "RChunker",
    "MarkdownChunker",
    "TextChunker",
    "CSVChunker",

    # Embeddings
    "OllamaEmbeddings",
    "CachedEmbeddings",
    "check_ollama_status",
    "normalize_embeddings",

    # Indexer
    "DocumentIndexer",
    "quick_index",

    # Batch Ingestion (S119)
    "BatchIngestEngine",
    "get_batch_ingest_engine",
    "scan_folder",
    "JobStatus",
    "FileStatus",
    "IngestJobRecord",
    "IngestFileRecord",

    # Retriever
    "DocumentRetriever",
    "SearchResult",
    "quick_search",
    "format_results",

    # Augmenter
    "PromptAugmenter",
    "AugmentedPrompt",
    "ContexteurRAGIntegration",
    "quick_augment",
]
