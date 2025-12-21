#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EMBEDDINGS - Interface avec Ollama pour les embeddings
======================================================
Génère des embeddings vectoriels via Ollama.

Modèles supportés :
- mxbai-embed-large (1024 dim, meilleure qualité)
- nomic-embed-text (768 dim, plus rapide)
"""

import logging
from typing import List, Optional
import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .config import get_config, EmbeddingConfig

logger = logging.getLogger(__name__)


class OllamaEmbeddings:
    """
    Client pour générer des embeddings via Ollama.
    
    Usage:
        embedder = OllamaEmbeddings()
        vectors = embedder.embed(["texte 1", "texte 2"])
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialise le client Ollama.
        
        Args:
            config: Configuration des embeddings (optionnel)
        """
        self.config = config or get_config().embedding
        self.url = f"{self.config.ollama_url}/api/embed"
        self._model_verified = False
        
    def _verify_model(self) -> bool:
        """Vérifie que le modèle d'embedding est disponible."""
        if self._model_verified:
            return True
            
        try:
            list_url = f"{self.config.ollama_url}/api/tags"
            response = requests.get(list_url, timeout=10)
            response.raise_for_status()
            
            models = response.json().get("models", [])
            model_names = [m.get("name", "").split(":")[0] for m in models]
            
            # Vérifier le modèle principal
            main_model = self.config.model.split(":")[0]
            if main_model in model_names or self.config.model in [m.get("name") for m in models]:
                self._model_verified = True
                logger.info(f"Modèle d'embedding vérifié: {self.config.model}")
                return True
            
            # Essayer le modèle alternatif
            fast_model = self.config.fast_model.split(":")[0]
            if fast_model in model_names:
                logger.warning(f"Modèle {self.config.model} non trouvé, utilisation de {self.config.fast_model}")
                self.config.model = self.config.fast_model
                self._model_verified = True
                return True
            
            logger.error(f"Aucun modèle d'embedding trouvé. Modèles disponibles: {model_names}")
            logger.error(f"Installez avec: ollama pull {self.config.model}")
            return False
            
        except requests.exceptions.ConnectionError:
            logger.error(f"Impossible de se connecter à Ollama ({self.config.ollama_url})")
            logger.error("Assurez-vous qu'Ollama est lancé avec: ollama serve")
            return False
        except Exception as e:
            logger.error(f"Erreur de vérification du modèle: {e}")
            return False
    
    def embed_single(self, text: str) -> Optional[List[float]]:
        """
        Génère l'embedding pour un texte unique.
        
        Args:
            text: Le texte à encoder
            
        Returns:
            Vecteur d'embedding ou None en cas d'erreur
        """
        if not self._verify_model():
            return None
            
        try:
            payload = {
                "model": self.config.model,
                "input": text
            }
            
            response = requests.post(
                self.url,
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            embeddings = result.get("embeddings", [])
            
            if embeddings:
                return embeddings[0]
            else:
                logger.warning("Aucun embedding retourné")
                return None
                
        except requests.exceptions.Timeout:
            logger.error(f"Timeout lors de l'embedding (>{self.config.timeout}s)")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur HTTP: {e}")
            return None
        except Exception as e:
            logger.error(f"Erreur d'embedding: {e}")
            return None
    
    def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Génère les embeddings pour un batch de textes.
        
        Ollama supporte le batch nativement avec le paramètre "input".
        
        Args:
            texts: Liste de textes à encoder
            
        Returns:
            Liste de vecteurs d'embedding
        """
        if not self._verify_model():
            return [None] * len(texts)
        
        if not texts:
            return []
            
        try:
            payload = {
                "model": self.config.model,
                "input": texts
            }
            
            response = requests.post(
                self.url,
                json=payload,
                timeout=self.config.timeout * 2  # Plus de temps pour les batches
            )
            response.raise_for_status()
            
            result = response.json()
            embeddings = result.get("embeddings", [])
            
            # S'assurer qu'on a le bon nombre de résultats
            if len(embeddings) != len(texts):
                logger.warning(f"Nombre d'embeddings ({len(embeddings)}) != nombre de textes ({len(texts)})")
            
            return embeddings
            
        except requests.exceptions.Timeout:
            logger.warning("Timeout sur le batch, passage en mode séquentiel")
            return self._embed_sequential(texts)
        except Exception as e:
            logger.error(f"Erreur batch, passage en mode séquentiel: {e}")
            return self._embed_sequential(texts)
    
    def _embed_sequential(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Fallback : embedding séquentiel en cas d'échec du batch."""
        results = []
        for text in texts:
            emb = self.embed_single(text)
            results.append(emb)
        return results
    
    def embed(
        self, 
        texts: List[str], 
        show_progress: bool = True,
        batch_size: Optional[int] = None
    ) -> List[Optional[List[float]]]:
        """
        Génère les embeddings pour une liste de textes.
        
        Traite les textes par batches pour optimiser les performances.
        
        Args:
            texts: Liste de textes à encoder
            show_progress: Afficher une barre de progression
            batch_size: Taille des batches (défaut: config.batch_size)
            
        Returns:
            Liste de vecteurs d'embedding
        """
        if not texts:
            return []
        
        if not self._verify_model():
            return [None] * len(texts)
        
        batch_size = batch_size or self.config.batch_size
        all_embeddings = []
        
        # Créer les batches
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        
        if show_progress:
            batches = tqdm(batches, desc="Embedding", unit="batch")
        
        for batch in batches:
            embeddings = self.embed_batch(batch)
            all_embeddings.extend(embeddings)
        
        return all_embeddings
    
    def get_dimension(self) -> int:
        """Retourne la dimension des embeddings du modèle."""
        # Dimensions connues par modèle
        dimensions = {
            "mxbai-embed-large": 1024,
            "nomic-embed-text": 768,
            "all-minilm": 384,
            "bge-m3": 1024,
        }
        
        model_name = self.config.model.split(":")[0]
        if model_name in dimensions:
            return dimensions[model_name]
        
        # Détecter dynamiquement
        test_emb = self.embed_single("test")
        if test_emb:
            return len(test_emb)
        
        return self.config.dimension


class CachedEmbeddings:
    """
    Wrapper avec cache pour éviter de recalculer les embeddings.
    
    Utilise un hash du contenu pour identifier les textes déjà vus.
    """
    
    def __init__(self, embedder: Optional[OllamaEmbeddings] = None):
        """
        Args:
            embedder: Instance OllamaEmbeddings (crée une nouvelle si non fourni)
        """
        self.embedder = embedder or OllamaEmbeddings()
        self._cache = {}
    
    def _hash_text(self, text: str) -> str:
        """Génère un hash pour le texte."""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()
    
    def embed(
        self, 
        texts: List[str],
        show_progress: bool = True
    ) -> List[Optional[List[float]]]:
        """
        Génère les embeddings avec cache.
        
        Args:
            texts: Liste de textes
            show_progress: Afficher la progression
            
        Returns:
            Liste d'embeddings
        """
        results = [None] * len(texts)
        texts_to_embed = []
        indices_to_embed = []
        
        # Vérifier le cache
        for i, text in enumerate(texts):
            text_hash = self._hash_text(text)
            if text_hash in self._cache:
                results[i] = self._cache[text_hash]
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)
        
        cache_hits = len(texts) - len(texts_to_embed)
        if cache_hits > 0:
            logger.info(f"Cache hits: {cache_hits}/{len(texts)}")
        
        # Calculer les nouveaux embeddings
        if texts_to_embed:
            new_embeddings = self.embedder.embed(texts_to_embed, show_progress)
            
            for i, (text, emb) in enumerate(zip(texts_to_embed, new_embeddings)):
                original_idx = indices_to_embed[i]
                results[original_idx] = emb
                
                # Mettre en cache
                if emb is not None:
                    text_hash = self._hash_text(text)
                    self._cache[text_hash] = emb
        
        return results
    
    def clear_cache(self):
        """Vide le cache."""
        self._cache.clear()
    
    @property
    def cache_size(self) -> int:
        """Nombre d'embeddings en cache."""
        return len(self._cache)


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def check_ollama_status() -> dict:
    """
    Vérifie le statut d'Ollama et des modèles d'embedding.
    
    Returns:
        Dictionnaire avec le statut
    """
    config = get_config().embedding
    status = {
        "ollama_running": False,
        "embedding_model_available": False,
        "model_name": config.model,
        "available_models": [],
        "error": None
    }
    
    try:
        # Vérifier qu'Ollama répond
        response = requests.get(f"{config.ollama_url}/api/tags", timeout=5)
        response.raise_for_status()
        status["ollama_running"] = True
        
        # Lister les modèles
        models = response.json().get("models", [])
        status["available_models"] = [m.get("name") for m in models]
        
        # Vérifier le modèle d'embedding
        model_base = config.model.split(":")[0]
        for m in models:
            if model_base in m.get("name", ""):
                status["embedding_model_available"] = True
                break
        
    except requests.exceptions.ConnectionError:
        status["error"] = "Ollama n'est pas accessible. Lancez: ollama serve"
    except Exception as e:
        status["error"] = str(e)
    
    return status


def normalize_embeddings(embeddings: List[List[float]]) -> np.ndarray:
    """
    Normalise les embeddings (L2 normalization).
    
    Args:
        embeddings: Liste de vecteurs
        
    Returns:
        Array numpy normalisé
    """
    arr = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Éviter division par zéro
    return arr / norms


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=== Test des embeddings Ollama ===\n")
    
    # Vérifier le statut
    status = check_ollama_status()
    print(f"Ollama actif: {status['ollama_running']}")
    print(f"Modèle embedding: {status['model_name']}")
    print(f"Modèle disponible: {status['embedding_model_available']}")
    
    if status['error']:
        print(f"Erreur: {status['error']}")
    else:
        print(f"Modèles disponibles: {len(status['available_models'])}")
    
    if status['embedding_model_available']:
        print("\n--- Test d'embedding ---")
        embedder = OllamaEmbeddings()
        
        texts = [
            "Fonction R pour calculer l'indice de Shannon",
            "Comment faire une PCA avec vegan en R",
            "Analyse de la diversité des orthoptères"
        ]
        
        embeddings = embedder.embed(texts, show_progress=False)
        
        for text, emb in zip(texts, embeddings):
            if emb:
                print(f"'{text[:40]}...' -> dim={len(emb)}")
            else:
                print(f"'{text[:40]}...' -> ERREUR")
        
        # Test de similarité
        if all(embeddings):
            embeddings_norm = normalize_embeddings(embeddings)
            similarity = np.dot(embeddings_norm[0], embeddings_norm[1])
            print(f"\nSimilarité texte 1-2: {similarity:.3f}")
