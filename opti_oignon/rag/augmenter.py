#!/usr/bin/env python3
"""
AUGMENTER - Augmentation des prompts avec contexte RAG
======================================================
Enriches queries with relevant retrieved context.

Features:
- Augmented prompt generation
- Format adapted to question type
- Integration with Contexteur 2.0
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .retriever import DocumentRetriever, SearchResult

logger = logging.getLogger(__name__)

# S144: Optional import of RAG sanitizer for injection defense
try:
    from opti_oignon.rag_sanitizer import (
        RAGSanitizer,
        SanitizationResult,
        get_rag_sanitizer,
    )
    RAG_SANITIZER_AVAILABLE = True
except ImportError:
    RAG_SANITIZER_AVAILABLE = False
    logger.debug("rag_sanitizer not available — injection defense disabled")


@dataclass
class AugmentedPrompt:
    """Prompt augmented with RAG context."""

    original_query: str           # Question originale
    augmented_prompt: str         # Prompt complet avec contexte
    context_chunks: list[SearchResult]  # Chunks used
    total_context_chars: int      # Context size

    @property
    def has_context(self) -> bool:
        """Indicate whether context was found."""
        return len(self.context_chunks) > 0

    @property
    def sources_summary(self) -> str:
        """Summary of sources used."""
        if not self.context_chunks:
            return "Aucune source"

        sources = set()
        for chunk in self.context_chunks:
            sources.add(chunk.source_name)

        return ", ".join(sorted(sources))


class PromptAugmenter:
    """
    Augments prompts with RAG context.

    Usage:
        augmenter = PromptAugmenter()
        result = augmenter.augment("Comment calculer l'indice de Shannon en R?")
        print(result.augmented_prompt)
    """

    # Prompt templates for different question types
    TEMPLATES = {
        "code": '''You are an expert programming assistant. Use the provided context to help as best you can.

## Context (extracted from my personal documents)

{context}

## Question

{query}

## Instructions
- Base your answer on the provided context when relevant
- If the context contains code, adapt it to the question
- Clearly indicate if you are using context information or general knowledge
- Provide commented code and explanations
''',

        "analysis": '''You are an expert assistant in data analysis and bioinformatics.

## Context (extracted from my personal documents)

{context}

## Question

{query}

## Instructions
- Use the context to understand my usual methods
- Propose solutions consistent with my practices
- Explique le raisonnement statistique
''',

        "general": '''## Contexte pertinent (de mes documents)

{context}

---

## Question

{query}

---

Use the above context to enrich your answer if relevant. If the context is not useful, respond normally.
''',

        "minimal": '''{query}

---
Contexte disponible :
{context}
''',
    }

    def __init__(
        self,
        retriever: DocumentRetriever | None = None,
        max_context_chars: int = 8000
    ):
        """
        Initialize the augmenter.

        Args:
            retriever: Retriever instance (creates new one if not provided)
            max_context_chars: Maximum context size in characters
        """
        self.retriever = retriever or DocumentRetriever()
        self.max_context_chars = max_context_chars

    def augment(
        self,
        query: str,
        n_results: int = 5,
        min_score: float = 0.3,
        template: str = "general",
        file_types: list[str] | None = None,
        include_sources: bool = True
    ) -> AugmentedPrompt:
        """
        Augment a query with RAG context.

        Args:
            query: Question de l'utilisateur
            n_results: Number of chunks to retrieve
            min_score: Score minimum de pertinence
            template: Type de template (code, analysis, general, minimal)
            file_types: File types to search
            include_sources: Include source references

        Returns:
            AugmentedPrompt avec le prompt enrichi
        """
        # Search for context
        results = self.retriever.search(
            query,
            n_results=n_results,
            min_score=min_score,
            file_types=file_types
        )

        # Build context
        context_parts = []
        total_chars = 0
        used_chunks = []

        for result in results:
            # Formater le chunk
            chunk_text = self._format_chunk(result, include_sources)

            # Check the size limit
            if total_chars + len(chunk_text) > self.max_context_chars:
                break

            context_parts.append(chunk_text)
            total_chars += len(chunk_text)
            used_chunks.append(result)

        # Construire le prompt final
        if context_parts:
            context = "\n\n".join(context_parts)
            template_text = self.TEMPLATES.get(template, self.TEMPLATES["general"])
            augmented = template_text.format(context=context, query=query)
        else:
            # No context found
            augmented = query

        return AugmentedPrompt(
            original_query=query,
            augmented_prompt=augmented,
            context_chunks=used_chunks,
            total_context_chars=total_chars
        )

    def _format_chunk(self, result: SearchResult, include_sources: bool) -> str:
        """Formate un chunk pour l'inclusion dans le contexte."""
        lines = []

        if include_sources:
            # Header avec source
            lines.append(f"### Source: {result.location}")
            lines.append(f"Type: {result.file_type} | Score: {result.score:.0%}")
            lines.append("")

        # Contenu
        lines.append(result.content)

        return "\n".join(lines)

    def augment_secure(
        self,
        query: str,
        system_prompt: str = "",
        *,
        n_results: int = 5,
        min_score: float = 0.3,
        file_types: list[str] | None = None,
        collection: str = "",
        sanitizer: "RAGSanitizer | None" = None,
    ) -> tuple["AugmentedPrompt", "SanitizationResult | None"]:
        """Augment a query with RAG context and injection defense (S144).

        Retrieves chunks, sanitizes them through the injection defense
        pipeline, and wraps the prompt with separation markers.

        Parameters
        ----------
        query : str
            User's query.
        system_prompt : str
            System-level instructions for the LLM.
        n_results : int
            Number of chunks to retrieve.
        min_score : float
            Minimum relevance score.
        file_types : list[str] or None
            Filter by file type.
        collection : str
            Collection name (for trust level resolution).
        sanitizer : RAGSanitizer or None
            Custom sanitizer instance. Uses singleton if None.

        Returns
        -------
        tuple[AugmentedPrompt, SanitizationResult | None]
            The augmented prompt and sanitization result (None if
            sanitizer not available).
        """
        # Retrieve chunks normally
        results = self.retriever.search(
            query, n_results=n_results, min_score=min_score,
            file_types=file_types,
        )

        if not results:
            return AugmentedPrompt(
                original_query=query,
                augmented_prompt=query if not system_prompt else f"{system_prompt}\n\n{query}",
                context_chunks=[],
                total_context_chars=0,
            ), None

        # If sanitizer not available, fall back to normal augmentation
        if not RAG_SANITIZER_AVAILABLE:
            logger.debug("RAG sanitizer not available, using unsecured augmentation")
            return self.augment(query, n_results=n_results, min_score=min_score), None

        # Prepare chunks for sanitization
        san = sanitizer or get_rag_sanitizer()
        chunk_dicts = []
        total_chars = 0
        used_results = []

        for result in results:
            text = result.content
            if total_chars + len(text) > self.max_context_chars:
                break
            chunk_dicts.append({
                "text": text,
                "chunk_id": f"{result.source_name}:{result.chunk_index}",
                "source": result.source_file or result.source_name,
                "collection": collection,
            })
            total_chars += len(text)
            used_results.append(result)

        # Run sanitization pipeline
        san_result = san.sanitize_chunks(chunk_dicts, collection=collection)

        # Build prompt with separation markers using safe chunks
        safe_chunks = san_result.safe_chunks
        wrapped = san.wrap_prompt(
            system_prompt=system_prompt or "You are a helpful assistant.",
            user_query=query,
            chunks=safe_chunks,
        )

        return AugmentedPrompt(
            original_query=query,
            augmented_prompt=wrapped,
            context_chunks=used_results,
            total_context_chars=sum(len(c.sanitized_text) for c in safe_chunks),
        ), san_result

    def detect_query_type(self, query: str) -> str:
        """
        Detect the question type to choose the template.

        Args:
            query: Question de l'utilisateur

        Returns:
            Recommended template type
        """
        query_lower = query.lower()

        # Keywords for code
        code_keywords = [
            "code", "fonction", "function", "script", "erreur", "bug",
            "import", "library", "package", "class", "def ", "r ", "python",
            "how to", "how to create", "write", "program"
        ]

        # Keywords for analysis
        analysis_keywords = [
            "analyse", "analyser", "statistique", "test", "pca", "nmds",
            "correlation", "regression", "glm", "gam", "diversity",
            "shannon", "simpson", "beta", "alpha", "permanova"
        ]

        # Detect the type
        for kw in code_keywords:
            if kw in query_lower:
                return "code"

        for kw in analysis_keywords:
            if kw in query_lower:
                return "analysis"

        return "general"

    def augment_smart(
        self,
        query: str,
        n_results: int = 5,
        min_score: float = 0.3,
        file_types: list[str] | None = None
    ) -> AugmentedPrompt:
        """
        Smart augmentation with automatic type detection.

        Args:
            query: Question de l'utilisateur
            n_results: Nombre de chunks
            min_score: Score minimum
            file_types: Types de fichiers

        Returns:
            Optimized AugmentedPrompt
        """
        template = self.detect_query_type(query)
        logger.debug(f"Detected question type: {template}")

        return self.augment(
            query,
            n_results=n_results,
            min_score=min_score,
            template=template,
            file_types=file_types
        )

    def get_context_only(
        self,
        query: str,
        n_results: int = 5,
        min_score: float = 0.3
    ) -> str:
        """
        Return only the formatted context (without template).

        Useful for inspection or manual integration.

        Args:
            query: Question de search
            n_results: Number of results
            min_score: Score minimum

        Returns:
            Formatted context
        """
        results = self.retriever.search(query, n_results=n_results, min_score=min_score)

        if not results:
            return "No relevant context found."

        parts = []
        for i, r in enumerate(results, 1):
            parts.append(f"### [{i}] {r.location} (score: {r.score:.0%})")
            parts.append(r.content)
            parts.append("")

        return "\n".join(parts)


# =============================================================================
# CONTEXTEUR 2.0 INTEGRATION
# =============================================================================

class ContexteurRAGIntegration:
    """
    Interface for integration with Opti-Oignon UI.

    Provides methods adapted for use in the Gradio interface.
    """

    def __init__(self):
        """Initialize the integration."""
        from .indexer import DocumentIndexer
        from .retriever import DocumentRetriever

        self.indexer = DocumentIndexer()
        self.retriever = DocumentRetriever()
        self.augmenter = PromptAugmenter(retriever=self.retriever)
        self._enabled = True

    @property
    def enabled(self) -> bool:
        """RAG enabled or not."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value

    def index_folder(
        self,
        folder_path: str,
        recursive: bool = True,
        force: bool = False,
    ) -> dict[str, Any]:
        """
        Index all supported files in a folder.

        Args:
            folder_path: Path to folder
            recursive: Index subfolders
            force: Force re-indexing

        Returns:
            Result with files_processed and chunks_created counts
        """
        result = self.indexer.index_directory(
            Path(folder_path),
            recursive=recursive,
            force=force
        )
        return {
            "files_processed": result.get("indexed_files", 0),
            "files_skipped": result.get("skipped_files", 0),
            "chunks_created": result.get("total_chunks", 0),
            "errors": result.get("errors", 0),
        }

    def search(
        self,
        query: str,
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Search for relevant documents.

        Args:
            query: Search query
            n_results: Maximum results to return

        Returns:
            List of results with content, source, and score
        """
        results = self.retriever.search(query, n_results=n_results)

        # Convert SearchResult objects to dicts for UI compatibility
        return [
            {
                "content": r.content,
                "source_file": r.source_file,
                "filename": r.source_name,
                "score": r.score,
                "chunk_index": r.chunk_index,
                "section_name": r.section_name,
                "file_type": r.file_type,
            }
            for r in results
        ]

    def enrich_query(
        self,
        query: str,
        n_results: int = 3,
    ) -> dict[str, Any]:
        """
        Enrich a query with relevant context from indexed documents.

        Args:
            query: User's query
            n_results: Number of context chunks to include

        Returns:
            Dictionary with enriched_prompt and sources
        """
        result = self.augmenter.augment_smart(query, n_results=n_results)

        sources = [
            {
                "file": r.source_name,
                "source_file": r.source_file,
                "section": r.section_name,
                "score": r.score,
                "type": r.file_type,
                "location": r.location,
            }
            for r in result.context_chunks
        ]

        return {
            "enriched_prompt": result.augmented_prompt,
            "sources": sources,
            "context_added": result.has_context,
            "original_query": query,
            "context_size": result.total_context_chars,
        }

    def enrich_prompt(
        self,
        query: str,
        use_rag: bool = True,
        n_chunks: int = 3,
        file_types: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Enriches a prompt for the Contexteur (legacy method).

        Args:
            query: User's question
            use_rag: Use RAG
            n_chunks: Number of chunks
            file_types: File types

        Returns:
            Dict with enriched prompt and metadata
        """
        if not use_rag or not self._enabled:
            return {
                "prompt": query,
                "rag_used": False,
                "sources": [],
                "context_size": 0
            }

        result = self.augmenter.augment_smart(
            query,
            n_results=n_chunks,
            file_types=file_types
        )

        sources = [
            {
                "file": r.source_name,
                "section": r.section_name,
                "score": r.score,
                "type": r.file_type
            }
            for r in result.context_chunks
        ]

        return {
            "prompt": result.augmented_prompt,
            "rag_used": result.has_context,
            "sources": sources,
            "context_size": result.total_context_chars,
            "original_query": query
        }

    def get_sources_display(self, sources: list[dict]) -> str:
        """
        Format sources for display in Gradio.

        Args:
            sources: List of sources

        Returns:
            Formatted text for display
        """
        if not sources:
            return "📭 No RAG sources used"

        lines = ["📚 **RAG Sources Used:**"]
        for s in sources:
            score_bar = "█" * int(s['score'] * 10) + "░" * (10 - int(s['score'] * 10))
            lines.append(f"  • `{s['file']}` ({s.get('type', 'unknown')}) [{score_bar}] {s['score']:.0%}")
            if s.get('section'):
                lines.append(f"    ↳ {s['section']}")

        return "\n".join(lines)

    def search_preview(self, query: str, n_results: int = 3) -> str:
        """
        Preview search results.

        Args:
            query: Query
            n_results: Number of results

        Returns:
            Formatted text for preview
        """
        results = self.retriever.search(query, n_results=n_results)

        if not results:
            return "🔍 No results found for this query."

        lines = [f"🔍 **{len(results)} result(s) found:**\n"]

        for i, r in enumerate(results, 1):
            preview = r.content[:150].replace("\n", " ")
            if len(r.content) > 150:
                preview += "..."

            lines.append(f"**{i}. {r.source_name}** (score: {r.score:.0%})")
            lines.append(f"   Type: {r.file_type} | Section: {r.section_name or 'N/A'}")
            lines.append(f"   > {preview}")
            lines.append("")

        return "\n".join(lines)

    def get_stats(self) -> dict[str, Any]:
        """Return RAG system statistics."""
        stats = self.indexer.get_stats()
        return {
            "total_chunks": stats.get("total_chunks", 0),
            "total_files": stats.get("total_files", 0),
            "files_by_type": stats.get("files_by_type", {}),
            "collection_name": stats.get("collection_name", ""),
            "storage_path": stats.get("storage_path", ""),
            "embedding_model": stats.get("embedding_model", ""),
            "enabled": self._enabled,
        }

    def clear(self) -> bool:
        """Clear all indexed documents."""
        return self.indexer.clear_index()


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def quick_augment(query: str) -> str:
    """
    Quick augmentation of a query.

    Args:
        query: Question

    Returns:
        Augmented prompt
    """
    augmenter = PromptAugmenter()
    result = augmenter.augment_smart(query)
    return result.augmented_prompt


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    augmenter = PromptAugmenter()

    print("=== Test de l'augmenter ===\n")

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "Comment calculer l'indice de Shannon en R?"

    print(f"Question: {query}\n")

    # Detect the type
    query_type = augmenter.detect_query_type(query)
    print(f"Detected type: {query_type}\n")

    # Augmenter
    result = augmenter.augment_smart(query)

    print(f"Context found: {result.has_context}")
    print(f"Sources: {result.sources_summary}")
    print(f"Context size: {result.total_context_chars} characters")
    print(f"\n{'='*60}")
    print("AUGMENTED PROMPT:")
    print('='*60)
    print(result.augmented_prompt)
