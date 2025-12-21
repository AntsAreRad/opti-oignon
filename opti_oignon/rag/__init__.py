#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

__version__ = "2.0.0"
__author__ = "Léon"

# Main imports
from .config import (
    get_config,
    set_config,
    RAGConfig,
    ChunkingConfig,
    EmbeddingConfig,
    RetrieverConfig,
)

from .chunkers import (
    Chunk,
    get_chunker,
    BaseChunker,
    CodeChunker,
    RChunker,
    MarkdownChunker,
    TextChunker,
    CSVChunker,
)

from .embeddings import (
    OllamaEmbeddings,
    CachedEmbeddings,
    check_ollama_status,
    normalize_embeddings,
)

from .indexer import (
    DocumentIndexer,
    quick_index,
)

from .retriever import (
    DocumentRetriever,
    SearchResult,
    quick_search,
    format_results,
)

from .augmenter import (
    PromptAugmenter,
    AugmentedPrompt,
    ContexteurRAGIntegration,
    quick_augment,
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
