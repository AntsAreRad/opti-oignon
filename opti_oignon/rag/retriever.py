#!/usr/bin/env python3
"""
RETRIEVER - Semantic search in documents
====================================================
Searches for the most relevant chunks for a query.

Features:
- Semantic similarity search
- Filtrage par type de fichier
- Filtrage par source
- Normalized relevance score
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from .config import RAGConfig, get_config
from .embeddings import OllamaEmbeddings

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Search result."""

    content: str                      # Contenu du chunk
    score: float                      # Similarity score (0-1)
    source_file: str                  # Source file path
    file_type: str                    # Type de fichier
    chunk_index: int                  # Index du chunk
    section_name: str | None       # Nom de la section
    start_line: int | None         # Start line
    end_line: int | None           # Ligne de fin
    metadata: dict[str, Any]          # Raw metadata

    @property
    def source_name(self) -> str:
        """Short name of the source file."""
        return Path(self.source_file).name

    @property
    def location(self) -> str:
        """Description de la localisation dans le fichier."""
        if self.section_name:
            return f"{self.source_name} ({self.section_name})"
        elif self.start_line and self.end_line:
            return f"{self.source_name} (lignes {self.start_line}-{self.end_line})"
        else:
            return self.source_name

    def __str__(self) -> str:
        """Text representation."""
        preview = self.content[:100] + "..." if len(self.content) > 100 else self.content
        return f"[{self.score:.2f}] {self.location}\n{preview}"


class DocumentRetriever:
    """
    Search in indexed documents.

    Usage:
        retriever = DocumentRetriever()
        results = retriever.search("comment calculer l'indice de Shannon")
        for r in results:
            print(r.location, r.score)
    """

    def __init__(
        self,
        config: RAGConfig | None = None,
        collection_name: str = "documents"
    ):
        """
        Initialize the retriever.

        Args:
            config: Configuration RAG
            collection_name: Nom de la collection ChromaDB
        """
        self.config = config or get_config()
        self.collection_name = collection_name

        # Client ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.config.chroma_dir),
            settings=Settings(anonymized_telemetry=False)
        )

        # Retrieve collection (don't create if it doesn't exist)
        try:
            self.collection = self.chroma_client.get_collection(collection_name)
        except Exception:
            logger.warning(f"Collection '{collection_name}' not found. Creating...")
            self.collection = self.chroma_client.get_or_create_collection(
                name=collection_name
            )

        # Embedder for queries
        self.embedder = OllamaEmbeddings(self.config.embedding)

    def search(
        self,
        query: str,
        n_results: int = 5,
        min_score: float | None = None,
        file_types: list[str] | None = None,
        source_files: list[str] | None = None,
        exclude_files: list[str] | None = None
    ) -> list[SearchResult]:
        """
        Search for the most relevant chunks.

        Args:
            query: Search query
            n_results: Maximum number of results
            min_score: Score minimum (0-1)
            file_types: File types to include
            source_files: Specific files to search
            exclude_files: Files to exclude

        Returns:
            List of SearchResult sorted by relevance
        """
        if not query.strip():
            return []

        # Check that there are documents
        if self.collection.count() == 0:
            logger.warning("No indexed documents")
            return []

        # Generate the query embedding
        query_embedding = self.embedder.embed_single(query)
        if query_embedding is None:
            logger.error("Unable to generate query embedding")
            return []

        # Construire les filtres
        where_filter = self._build_filter(file_types, source_files, exclude_files)

        # ChromaDB query
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results * 2, 20),  # Prendre plus pour filtrer ensuite
                where=where_filter if where_filter else None,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            logger.error(f"Erreur search: {e}")
            return []

        # Convert results
        search_results = []

        if not results or not results.get('documents'):
            return []

        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            # Convert distance to score (ChromaDB returns distances)
            # For normalized embeddings, distance = 2 - 2*cosine_similarity
            # Donc score = 1 - distance/2
            score = max(0, 1 - dist / 2)

            # Filtrer par score minimum
            min_s = min_score if min_score is not None else self.config.retriever.min_score
            if score < min_s:
                continue

            result = SearchResult(
                content=doc,
                score=score,
                source_file=meta.get("source_file", ""),
                file_type=meta.get("file_type", ""),
                chunk_index=meta.get("chunk_index", 0),
                section_name=meta.get("section_name"),
                start_line=meta.get("start_line"),
                end_line=meta.get("end_line"),
                metadata=meta
            )

            search_results.append(result)

        # Sort by descending score
        search_results.sort(key=lambda x: x.score, reverse=True)

        # Limit to requested count
        return search_results[:n_results]

    def _build_filter(
        self,
        file_types: list[str] | None,
        source_files: list[str] | None,
        exclude_files: list[str] | None
    ) -> dict | None:
        """Build the ChromaDB filter."""
        conditions = []

        # Filtre par type
        if file_types:
            if len(file_types) == 1:
                conditions.append({"file_type": {"$eq": file_types[0]}})
            else:
                conditions.append({"file_type": {"$in": file_types}})

        # Filtre par fichier source
        if source_files:
            if len(source_files) == 1:
                conditions.append({"source_file": {"$eq": source_files[0]}})
            else:
                conditions.append({"source_file": {"$in": source_files}})

        # Exclusion de fichiers
        if exclude_files:
            for f in exclude_files:
                conditions.append({"source_file": {"$ne": f}})

        # Combiner les conditions
        if not conditions:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}

    def search_similar(
        self,
        reference_text: str,
        n_results: int = 5,
        exclude_self: bool = True
    ) -> list[SearchResult]:
        """
        Find chunks similar to a reference text.

        Useful for finding similar code or related passages.

        Args:
            reference_text: Reference text
            n_results: Number of results
            exclude_self: Exclure le texte exact

        Returns:
            List of similar results
        """
        results = self.search(reference_text, n_results=n_results + 5)

        if exclude_self:
            # Filtrer les correspondances exactes
            results = [r for r in results if r.content.strip() != reference_text.strip()]

        return results[:n_results]

    def search_by_file(
        self,
        filepath: str,
        query: str | None = None,
        n_results: int = 10
    ) -> list[SearchResult]:
        """
        Search in a specific file.

        Args:
            filepath: File path
            query: Optional query (otherwise returns all chunks)
            n_results: Number of results

        Returns:
            List of results
        """
        if query:
            return self.search(
                query,
                n_results=n_results,
                source_files=[str(Path(filepath).resolve())]
            )

        # Without query, return all chunks from the file
        try:
            results = self.collection.get(
                where={"source_file": str(Path(filepath).resolve())},
                include=["documents", "metadatas"]
            )

            if not results or not results.get('documents'):
                return []

            search_results = []
            for doc, meta in zip(results['documents'], results['metadatas']):
                result = SearchResult(
                    content=doc,
                    score=1.0,  # Score parfait car pas de search
                    source_file=meta.get("source_file", ""),
                    file_type=meta.get("file_type", ""),
                    chunk_index=meta.get("chunk_index", 0),
                    section_name=meta.get("section_name"),
                    start_line=meta.get("start_line"),
                    end_line=meta.get("end_line"),
                    metadata=meta
                )
                search_results.append(result)

            # Trier par index de chunk
            search_results.sort(key=lambda x: x.chunk_index)
            return search_results[:n_results]

        except Exception as e:
            logger.error(f"Erreur search fichier: {e}")
            return []

    def get_context_window(
        self,
        result: SearchResult,
        window_size: int = 1
    ) -> list[SearchResult]:
        """
        Retrieve chunks adjacent to a result.

        Useful for getting more context around a result.

        Args:
            result: Initial result
            window_size: Number of chunks before/after

        Returns:
            List of chunks including context
        """
        try:
            # Retrieve all chunks from the same file
            all_chunks = self.search_by_file(result.source_file, n_results=100)

            # Trouver l'index du chunk actuel
            current_idx = None
            for i, chunk in enumerate(all_chunks):
                if chunk.chunk_index == result.chunk_index:
                    current_idx = i
                    break

            if current_idx is None:
                return [result]

            # Extract the window
            start = max(0, current_idx - window_size)
            end = min(len(all_chunks), current_idx + window_size + 1)

            return all_chunks[start:end]

        except Exception as e:
            logger.error(f"Erreur contexte: {e}")
            return [result]

    def count(self) -> int:
        """Return the total number of indexed chunks."""
        return self.collection.count()

    def get_file_types(self) -> list[str]:
        """Return the list of indexed file types."""
        try:
            # Query to get unique metadata
            results = self.collection.get(include=["metadatas"])
            if not results or not results.get('metadatas'):
                return []

            types = set()
            for meta in results['metadatas']:
                ft = meta.get('file_type')
                if ft:
                    types.add(ft)

            return sorted(list(types))
        except Exception:
            return []


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def quick_search(query: str, n_results: int = 5) -> list[SearchResult]:
    """
    Fonction rapide de search.

    Args:
        query: Query
        n_results: Number of results

    Returns:
        List of results
    """
    retriever = DocumentRetriever()
    return retriever.search(query, n_results=n_results)


def format_results(results: list[SearchResult], show_content: bool = True) -> str:
    """
    Format results for display.

    Args:
        results: List of results
        show_content: Afficher le contenu

    Returns:
        Formatted text
    """
    if not results:
        return "No results found."

    output = []
    for i, r in enumerate(results, 1):
        output.append(f"\n{'='*60}")
        output.append(f"Result {i}/{len(results)} - Score: {r.score:.2%}")
        output.append(f"Source: {r.location}")
        output.append(f"Type: {r.file_type}")
        output.append('='*60)

        if show_content:
            output.append(r.content)

    return "\n".join(output)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    retriever = DocumentRetriever()

    print("=== Test du retriever ===\n")
    print(f"Indexed chunks: {retriever.count()}")
    print(f"Types de fichiers: {retriever.get_file_types()}")

    # Search interactive
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"\nRecherche: {query}")

        results = retriever.search(query, n_results=3)
        print(format_results(results))
    else:
        print("\nUsage: python retriever.py <query>")
