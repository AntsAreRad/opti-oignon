#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AUGMENTER - Augmentation des prompts avec contexte RAG
======================================================
Enrichit les requêtes avec le contexte pertinent récupéré.

Fonctionnalités :
- Génération de prompts augmentés
- Formatage adapté au type de question
- Intégration avec le Contexteur 2.0
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path

from .retriever import DocumentRetriever, SearchResult
from .config import get_config

logger = logging.getLogger(__name__)


@dataclass
class AugmentedPrompt:
    """Prompt augmenté avec contexte RAG."""
    
    original_query: str           # Question originale
    augmented_prompt: str         # Prompt complet avec contexte
    context_chunks: List[SearchResult]  # Chunks utilisés
    total_context_chars: int      # Taille du contexte
    
    @property
    def has_context(self) -> bool:
        """Indique si du contexte a été trouvé."""
        return len(self.context_chunks) > 0
    
    @property
    def sources_summary(self) -> str:
        """Résumé des sources utilisées."""
        if not self.context_chunks:
            return "Aucune source"
        
        sources = set()
        for chunk in self.context_chunks:
            sources.add(chunk.source_name)
        
        return ", ".join(sorted(sources))


class PromptAugmenter:
    """
    Augmente les prompts avec du contexte RAG.
    
    Usage:
        augmenter = PromptAugmenter()
        result = augmenter.augment("Comment calculer l'indice de Shannon en R?")
        print(result.augmented_prompt)
    """
    
    # Templates de prompt pour différents types de questions
    TEMPLATES = {
        "code": '''Tu es un assistant expert en programmation. Utilise le contexte fourni pour aider au mieux.

## Contexte (extrait de mes documents personnels)

{context}

## Question

{query}

## Instructions
- Base ta réponse sur le contexte fourni quand c'est pertinent
- Si le contexte contient du code, adapte-le à la question
- Indique clairement si tu utilises des informations du contexte ou tes connaissances générales
- Fournis du code commenté et des explications
''',
        
        "analysis": '''Tu es un assistant expert en analyse de données et bioinformatique.

## Contexte (extrait de mes documents personnels)

{context}

## Question

{query}

## Instructions
- Utilise le contexte pour comprendre mes méthodes habituelles
- Propose des solutions cohérentes avec mes pratiques
- Explique le raisonnement statistique
''',
        
        "general": '''## Contexte pertinent (de mes documents)

{context}

---

## Question

{query}

---

Utilise le contexte ci-dessus pour enrichir ta réponse si pertinent. Si le contexte n'est pas utile, réponds normalement.
''',

        "minimal": '''{query}

---
Contexte disponible :
{context}
''',
    }
    
    def __init__(
        self,
        retriever: Optional[DocumentRetriever] = None,
        max_context_chars: int = 8000
    ):
        """
        Initialise l'augmenter.
        
        Args:
            retriever: Instance du retriever (crée une nouvelle si non fourni)
            max_context_chars: Taille maximale du contexte en caractères
        """
        self.retriever = retriever or DocumentRetriever()
        self.max_context_chars = max_context_chars
    
    def augment(
        self,
        query: str,
        n_results: int = 5,
        min_score: float = 0.3,
        template: str = "general",
        file_types: Optional[List[str]] = None,
        include_sources: bool = True
    ) -> AugmentedPrompt:
        """
        Augmente une requête avec du contexte RAG.
        
        Args:
            query: Question de l'utilisateur
            n_results: Nombre de chunks à récupérer
            min_score: Score minimum de pertinence
            template: Type de template (code, analysis, general, minimal)
            file_types: Types de fichiers à chercher
            include_sources: Inclure les références des sources
            
        Returns:
            AugmentedPrompt avec le prompt enrichi
        """
        # Rechercher le contexte
        results = self.retriever.search(
            query,
            n_results=n_results,
            min_score=min_score,
            file_types=file_types
        )
        
        # Construire le contexte
        context_parts = []
        total_chars = 0
        used_chunks = []
        
        for result in results:
            # Formater le chunk
            chunk_text = self._format_chunk(result, include_sources)
            
            # Vérifier la limite de taille
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
            # Pas de contexte trouvé
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
    
    def detect_query_type(self, query: str) -> str:
        """
        Détecte le type de question pour choisir le template.
        
        Args:
            query: Question de l'utilisateur
            
        Returns:
            Type de template recommandé
        """
        query_lower = query.lower()
        
        # Mots-clés pour le code
        code_keywords = [
            "code", "fonction", "function", "script", "erreur", "bug",
            "import", "library", "package", "class", "def ", "r ", "python",
            "comment faire", "comment créer", "écrire", "programmer"
        ]
        
        # Mots-clés pour l'analyse
        analysis_keywords = [
            "analyse", "analyser", "statistique", "test", "pca", "nmds",
            "correlation", "regression", "glm", "gam", "diversité",
            "shannon", "simpson", "beta", "alpha", "permanova"
        ]
        
        # Détecter le type
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
        file_types: Optional[List[str]] = None
    ) -> AugmentedPrompt:
        """
        Augmentation intelligente avec détection automatique du type.
        
        Args:
            query: Question de l'utilisateur
            n_results: Nombre de chunks
            min_score: Score minimum
            file_types: Types de fichiers
            
        Returns:
            AugmentedPrompt optimisé
        """
        template = self.detect_query_type(query)
        logger.debug(f"Type de question détecté: {template}")
        
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
        Retourne uniquement le contexte formaté (sans template).
        
        Utile pour l'inspection ou l'intégration manuelle.
        
        Args:
            query: Question de recherche
            n_results: Nombre de résultats
            min_score: Score minimum
            
        Returns:
            Contexte formaté
        """
        results = self.retriever.search(query, n_results=n_results, min_score=min_score)
        
        if not results:
            return "Aucun contexte pertinent trouvé."
        
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(f"### [{i}] {r.location} (score: {r.score:.0%})")
            parts.append(r.content)
            parts.append("")
        
        return "\n".join(parts)


# =============================================================================
# INTÉGRATION CONTEXTEUR 2.0
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
    ) -> Dict[str, Any]:
        """
        Index all supported files in a folder.
        
        Args:
            folder_path: Path to folder
            recursive: Index subfolders
            force: Force re-indexing
            
        Returns:
            Result with files_processed and chunks_created counts
        """
        from pathlib import Path
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
    ) -> List[Dict[str, Any]]:
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
    ) -> Dict[str, Any]:
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
        file_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
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
    
    def get_sources_display(self, sources: List[Dict]) -> str:
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
    
    def get_stats(self) -> Dict[str, Any]:
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
    Augmentation rapide d'une requête.
    
    Args:
        query: Question
        
    Returns:
        Prompt augmenté
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
    
    # Détecter le type
    query_type = augmenter.detect_query_type(query)
    print(f"Type détecté: {query_type}\n")
    
    # Augmenter
    result = augmenter.augment_smart(query)
    
    print(f"Contexte trouvé: {result.has_context}")
    print(f"Sources: {result.sources_summary}")
    print(f"Taille contexte: {result.total_context_chars} caractères")
    print(f"\n{'='*60}")
    print("PROMPT AUGMENTÉ:")
    print('='*60)
    print(result.augmented_prompt)
