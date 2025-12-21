#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RETRIEVER - Recherche sémantique dans les documents
====================================================
Recherche les chunks les plus pertinents pour une requête.

Fonctionnalités :
- Recherche par similarité sémantique
- Filtrage par type de fichier
- Filtrage par source
- Score de pertinence normalisé
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

import chromadb
from chromadb.config import Settings

from .config import get_config, RAGConfig
from .embeddings import OllamaEmbeddings

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Résultat d'une recherche."""
    
    content: str                      # Contenu du chunk
    score: float                      # Score de similarité (0-1)
    source_file: str                  # Chemin du fichier source
    file_type: str                    # Type de fichier
    chunk_index: int                  # Index du chunk
    section_name: Optional[str]       # Nom de la section
    start_line: Optional[int]         # Ligne de début
    end_line: Optional[int]           # Ligne de fin
    metadata: Dict[str, Any]          # Métadonnées brutes
    
    @property
    def source_name(self) -> str:
        """Nom court du fichier source."""
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
        """Représentation textuelle."""
        preview = self.content[:100] + "..." if len(self.content) > 100 else self.content
        return f"[{self.score:.2f}] {self.location}\n{preview}"


class DocumentRetriever:
    """
    Recherche dans les documents indexés.
    
    Usage:
        retriever = DocumentRetriever()
        results = retriever.search("comment calculer l'indice de Shannon")
        for r in results:
            print(r.location, r.score)
    """
    
    def __init__(
        self, 
        config: Optional[RAGConfig] = None,
        collection_name: str = "documents"
    ):
        """
        Initialise le retriever.
        
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
        
        # Récupérer la collection (ne pas la créer si elle n'existe pas)
        try:
            self.collection = self.chroma_client.get_collection(collection_name)
        except Exception:
            logger.warning(f"Collection '{collection_name}' non trouvée. Création...")
            self.collection = self.chroma_client.get_or_create_collection(
                name=collection_name
            )
        
        # Embedder pour les requêtes
        self.embedder = OllamaEmbeddings(self.config.embedding)
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        min_score: Optional[float] = None,
        file_types: Optional[List[str]] = None,
        source_files: Optional[List[str]] = None,
        exclude_files: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """
        Recherche les chunks les plus pertinents.
        
        Args:
            query: Requête de recherche
            n_results: Nombre maximum de résultats
            min_score: Score minimum (0-1)
            file_types: Types de fichiers à inclure
            source_files: Fichiers spécifiques à chercher
            exclude_files: Fichiers à exclure
            
        Returns:
            Liste de SearchResult triée par pertinence
        """
        if not query.strip():
            return []
        
        # Vérifier qu'il y a des documents
        if self.collection.count() == 0:
            logger.warning("Aucun document indexé")
            return []
        
        # Générer l'embedding de la requête
        query_embedding = self.embedder.embed_single(query)
        if query_embedding is None:
            logger.error("Impossible de générer l'embedding de la requête")
            return []
        
        # Construire les filtres
        where_filter = self._build_filter(file_types, source_files, exclude_files)
        
        # Requête ChromaDB
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results * 2, 20),  # Prendre plus pour filtrer ensuite
                where=where_filter if where_filter else None,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            logger.error(f"Erreur recherche: {e}")
            return []
        
        # Convertir les résultats
        search_results = []
        
        if not results or not results.get('documents'):
            return []
        
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        
        for doc, meta, dist in zip(documents, metadatas, distances):
            # Convertir la distance en score (ChromaDB retourne des distances)
            # Pour les embeddings normalisés, distance = 2 - 2*cosine_similarity
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
        
        # Trier par score décroissant
        search_results.sort(key=lambda x: x.score, reverse=True)
        
        # Limiter au nombre demandé
        return search_results[:n_results]
    
    def _build_filter(
        self,
        file_types: Optional[List[str]],
        source_files: Optional[List[str]],
        exclude_files: Optional[List[str]]
    ) -> Optional[Dict]:
        """Construit le filtre ChromaDB."""
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
    ) -> List[SearchResult]:
        """
        Trouve les chunks similaires à un texte de référence.
        
        Utile pour trouver du code similaire ou des passages liés.
        
        Args:
            reference_text: Texte de référence
            n_results: Nombre de résultats
            exclude_self: Exclure le texte exact
            
        Returns:
            Liste de résultats similaires
        """
        results = self.search(reference_text, n_results=n_results + 5)
        
        if exclude_self:
            # Filtrer les correspondances exactes
            results = [r for r in results if r.content.strip() != reference_text.strip()]
        
        return results[:n_results]
    
    def search_by_file(
        self,
        filepath: str,
        query: Optional[str] = None,
        n_results: int = 10
    ) -> List[SearchResult]:
        """
        Recherche dans un fichier spécifique.
        
        Args:
            filepath: Chemin du fichier
            query: Requête optionnelle (sinon retourne tous les chunks)
            n_results: Nombre de résultats
            
        Returns:
            Liste de résultats
        """
        if query:
            return self.search(
                query,
                n_results=n_results,
                source_files=[str(Path(filepath).resolve())]
            )
        
        # Sans requête, retourner tous les chunks du fichier
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
                    score=1.0,  # Score parfait car pas de recherche
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
            logger.error(f"Erreur recherche fichier: {e}")
            return []
    
    def get_context_window(
        self,
        result: SearchResult,
        window_size: int = 1
    ) -> List[SearchResult]:
        """
        Récupère les chunks adjacents à un résultat.
        
        Utile pour avoir plus de contexte autour d'un résultat.
        
        Args:
            result: Résultat initial
            window_size: Nombre de chunks avant/après
            
        Returns:
            Liste de chunks incluant le contexte
        """
        try:
            # Récupérer tous les chunks du même fichier
            all_chunks = self.search_by_file(result.source_file, n_results=100)
            
            # Trouver l'index du chunk actuel
            current_idx = None
            for i, chunk in enumerate(all_chunks):
                if chunk.chunk_index == result.chunk_index:
                    current_idx = i
                    break
            
            if current_idx is None:
                return [result]
            
            # Extraire la fenêtre
            start = max(0, current_idx - window_size)
            end = min(len(all_chunks), current_idx + window_size + 1)
            
            return all_chunks[start:end]
            
        except Exception as e:
            logger.error(f"Erreur contexte: {e}")
            return [result]
    
    def count(self) -> int:
        """Retourne le nombre total de chunks indexés."""
        return self.collection.count()
    
    def get_file_types(self) -> List[str]:
        """Retourne la liste des types de fichiers indexés."""
        try:
            # Requête pour obtenir les métadonnées uniques
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

def quick_search(query: str, n_results: int = 5) -> List[SearchResult]:
    """
    Fonction rapide de recherche.
    
    Args:
        query: Requête
        n_results: Nombre de résultats
        
    Returns:
        Liste de résultats
    """
    retriever = DocumentRetriever()
    return retriever.search(query, n_results=n_results)


def format_results(results: List[SearchResult], show_content: bool = True) -> str:
    """
    Formate les résultats pour l'affichage.
    
    Args:
        results: Liste de résultats
        show_content: Afficher le contenu
        
    Returns:
        Texte formaté
    """
    if not results:
        return "Aucun résultat trouvé."
    
    output = []
    for i, r in enumerate(results, 1):
        output.append(f"\n{'='*60}")
        output.append(f"Résultat {i}/{len(results)} - Score: {r.score:.2%}")
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
    print(f"Chunks indexés: {retriever.count()}")
    print(f"Types de fichiers: {retriever.get_file_types()}")
    
    # Recherche interactive
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"\nRecherche: {query}")
        
        results = retriever.search(query, n_results=3)
        print(format_results(results))
    else:
        print("\nUsage: python retriever.py <requête>")
