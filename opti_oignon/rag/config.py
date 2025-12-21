#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG CONFIGURATION - Opti-Oignon RAG System
==========================================

Centralized configuration for the RAG system.
Adapted from the original Contexteur RAG.

Author: Léon
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os

# =============================================================================
# DEFAULT PATHS
# =============================================================================

# RAG storage folder (inside opti_oignon/data)
_MODULE_DIR = Path(__file__).parent
RAG_HOME = Path(os.environ.get("RAG_HOME", _MODULE_DIR.parent / "data" / "rag"))

# Sub-folders
CHROMA_DIR = RAG_HOME / "chroma_db"
CACHE_DIR = RAG_HOME / "cache"
LOGS_DIR = RAG_HOME / "logs"


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""
    
    # Maximum chunk size (in approximate tokens)
    max_chunk_size: int = 500
    
    # Overlap between chunks (in tokens)
    chunk_overlap: int = 50
    
    # Separators by file type
    code_separators: List[str] = field(default_factory=lambda: [
        "\n\n\n",      # Triple line break (section separation)
        "\ndef ",      # Python function definition
        "\nclass ",    # Python class definition
        "\n# ===",     # Section separator (comment)
        "\n# ---",     # Sub-section separator
    ])
    
    r_separators: List[str] = field(default_factory=lambda: [
        "\n\n\n",           # Triple line break
        "\n# ===",          # Section separator
        "\n# ---",          # Sub-section separator
        "\n\n# ",           # New comment block
    ])
    
    markdown_separators: List[str] = field(default_factory=lambda: [
        "\n## ",       # Header level 2
        "\n### ",      # Header level 3
        "\n#### ",     # Header level 4
        "\n---\n",     # Horizontal separator
        "\n\n\n",      # Triple line break
    ])
    
    text_separators: List[str] = field(default_factory=lambda: [
        "\n\n\n",      # Triple line break
        "\n\n",        # Double line break (paragraph)
        ". ",          # End of sentence
    ])


@dataclass
class EmbeddingConfig:
    """Configuration for embeddings."""
    
    # Ollama embedding model
    model: str = "mxbai-embed-large"
    
    # Alternative model (faster)
    fast_model: str = "nomic-embed-text"
    
    # Ollama URL
    ollama_url: str = "http://localhost:11434"
    
    # Embedding dimension (depends on model)
    # mxbai-embed-large: 1024
    # nomic-embed-text: 768
    dimension: int = 1024
    
    # Batch size for embedding
    batch_size: int = 32
    
    # Timeout in seconds
    timeout: int = 120


@dataclass
class RetrieverConfig:
    """Configuration for search."""
    
    # Default number of results
    n_results: int = 5
    
    # Minimum similarity score (0-1)
    min_score: float = 0.3
    
    # Default file types to include
    default_file_types: List[str] = field(default_factory=lambda: [
        "py", "r", "R", "md", "txt", "csv", "json", "yaml", "yml", "sh", "bash"
    ])


@dataclass
class RAGConfig:
    """Main RAG system configuration."""
    
    # Sub-module configurations
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    
    # Paths
    rag_home: Path = RAG_HOME
    chroma_dir: Path = CHROMA_DIR
    cache_dir: Path = CACHE_DIR
    logs_dir: Path = LOGS_DIR
    
    # Supported extensions with their type
    file_type_mapping: Dict[str, str] = field(default_factory=lambda: {
        # Python code
        ".py": "python",
        ".pyw": "python",
        ".pyi": "python",
        
        # R code
        ".r": "r",
        ".R": "r",
        ".Rmd": "rmarkdown",
        ".rmd": "rmarkdown",
        
        # Markdown and text
        ".md": "markdown",
        ".markdown": "markdown",
        ".txt": "text",
        ".text": "text",
        
        # Configuration
        ".yaml": "yaml",
        ".yml": "yaml",
        ".json": "json",
        ".toml": "toml",
        ".ini": "ini",
        ".cfg": "ini",
        
        # Shell
        ".sh": "shell",
        ".bash": "shell",
        ".zsh": "shell",
        
        # Data - Tabular
        ".csv": "csv",
        ".tsv": "csv",
        ".xlsx": "excel",
        ".xls": "excel",
        
        # Documents
        ".pdf": "pdf",
        ".docx": "docx",
        ".doc": "docx",
        
        # SQL
        ".sql": "sql",
        
        # Other languages
        ".js": "javascript",
        ".ts": "typescript",
        ".html": "html",
        ".css": "css",
    })
    
    # Files/folders to ignore
    ignore_patterns: List[str] = field(default_factory=lambda: [
        "__pycache__",
        ".git",
        ".svn",
        "node_modules",
        ".venv",
        "venv",
        ".env",
        "*.pyc",
        "*.pyo",
        ".DS_Store",
        "Thumbs.db",
        "*.egg-info",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    ])
    
    # Maximum file size to index (in MB)
    max_file_size_mb: float = 10.0
    
    def __post_init__(self):
        """Create necessary directories."""
        self.rag_home.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

# Default configuration (singleton)
_config: Optional[RAGConfig] = None


def get_config() -> RAGConfig:
    """Return the global configuration."""
    global _config
    if _config is None:
        _config = RAGConfig()
    return _config


def set_config(config: RAGConfig) -> None:
    """Set the global configuration."""
    global _config
    _config = config


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    config = get_config()
    print(f"RAG Home: {config.rag_home}")
    print(f"ChromaDB: {config.chroma_dir}")
    print(f"Embedding model: {config.embedding.model}")
    print(f"Chunk size: {config.chunking.max_chunk_size}")
    print(f"Supported types: {len(config.file_type_mapping)} extensions")
